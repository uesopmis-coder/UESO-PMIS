from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util


class GenerateTeamRequest(BaseModel):
    keywords: str = Field(..., min_length=1)
    include_in_progress: bool = False
    campus_filter: int | str | None = None
    college_filter: int | str | None = None
    num_participants: int = Field(5, ge=1, le=20)
    candidates: list[dict[str, Any]] = Field(default_factory=list)


class TeamScorer:
    DEGREE_WEIGHT = 0.15
    EXPERTISE_WEIGHT = 0.40
    PROJECT_WEIGHT = 0.40
    LEXICAL_WEIGHT = 0.05

    MIN_SEMANTIC_SCORE = 0.25
    MIN_FINAL_SCORE = 0.19
    FALLBACK_SEMANTIC_SCORE = 0.17
    FALLBACK_MIN_FINAL_SCORE = 0.14
    MIN_ANCHOR_SCORE = 0.23
    MIN_LEXICAL_MATCH_SCORE = 0.08
    PROFILE_MIN_SCORE = 0.15
    PROJECT_MIN_SCORE = 0.12
    DEGREE_MIN_SCORE = 0.16
    EXPERTISE_MIN_SCORE = 0.18

    MAX_PROJECT_TITLES_FOR_SCORING = 12
    TOP_PROJECT_TITLES_FOR_SCORE = 3
    MAX_WORKLOAD_PENALTY = 0.12

    STOP_WORDS = {
        'the', 'and', 'for', 'with', 'from', 'that', 'this', 'into', 'your', 'our',
        'using', 'through', 'about', 'have', 'has', 'are', 'was', 'were', 'will', 'can',
        'able', 'more', 'less', 'over', 'under', 'into', 'onto', 'a', 'an', 'of', 'to', 'in',
    }

    SHORT_DOMAIN_TOKENS = {'ai', 'it', 'ict', 'ml'}

    KEYWORD_EXPANSIONS = {
        'computer': {'computing', 'software', 'programming', 'technology', 'digital', 'information technology', 'information systems', 'coding'},
        'technology': {'tech', 'digital', 'software', 'computing', 'information systems', 'automation'},
        'software': {'application development', 'programming', 'coding', 'computer systems', 'it systems'},
        'data': {'analytics', 'analysis', 'database', 'information', 'statistics', 'data science'},
        'ai': {'artificial intelligence', 'machine learning', 'deep learning', 'nlp', 'predictive analytics'},
        'finance': {'financial', 'accounting', 'budget', 'economics', 'bookkeeping', 'fiscal management'},
        'education': {'teaching', 'learning', 'curriculum', 'training', 'instruction', 'student development'},
        'health': {'healthcare', 'medical', 'wellness', 'public health', 'clinical', 'nutrition'},
        'environment': {'sustainability', 'conservation', 'ecology', 'climate', 'biodiversity', 'marine'},
    }

    MODEL_NAME = 'all-MiniLM-L6-v2'
    MODEL_CACHE_DIR = os.path.join(os.getenv('HF_HOME', '/data'), 'ai_model')

    def __init__(self):
        self.model = None

    def _load_model(self):
        if self.model is None:
            os.makedirs(self.MODEL_CACHE_DIR, exist_ok=True)
            self.model = SentenceTransformer(self.MODEL_NAME, cache_folder=self.MODEL_CACHE_DIR)
        return self.model

    def _softmax(self, x):
        if len(x) == 0:
            return np.array([])
        x = np.array(x)
        exp = np.exp(x - np.max(x))
        return exp / exp.sum()

    def _normalize_text(self, text):
        if not text:
            return ''
        normalized = re.sub(r'[^a-z0-9\s]', ' ', str(text).lower())
        return re.sub(r'\s+', ' ', normalized).strip()

    def _tokenize(self, text):
        normalized = self._normalize_text(text)
        if not normalized:
            return set()
        return {
            token for token in normalized.split()
            if (len(token) > 2 or token in self.SHORT_DOMAIN_TOKENS) and token not in self.STOP_WORDS
        }

    def _expand_tokens(self, tokens):
        if not tokens:
            return set()

        expanded = set(tokens)
        for key, related_terms in self.KEYWORD_EXPANSIONS.items():
            concept_tokens = set()
            concept_tokens.update(self._tokenize(key))
            for term in related_terms:
                concept_tokens.update(self._tokenize(term))

            if tokens.intersection(concept_tokens):
                expanded.update(concept_tokens)

        return expanded

    def _expand_keywords(self, keyword_text):
        base_tokens = self._tokenize(keyword_text)
        expanded_tokens = self._expand_tokens(base_tokens)

        for key, related_terms in self.KEYWORD_EXPANSIONS.items():
            key_normalized = self._normalize_text(key)
            if ' ' in key_normalized and key_normalized in keyword_text:
                for term in related_terms:
                    expanded_tokens.update(self._tokenize(term))

        expanded_text = f"{keyword_text} {' '.join(sorted(expanded_tokens))}".strip()
        return expanded_text, expanded_tokens

    def _clamp_score(self, score):
        try:
            score_val = float(score)
        except (TypeError, ValueError):
            return 0.0
        if np.isnan(score_val):
            return 0.0
        return max(0.0, min(1.0, score_val))

    def _sim(self, emb1, emb2):
        return self._clamp_score(util.cos_sim(emb1, emb2)[0][0])

    def _semantic_score(self, keyword_emb, text, model):
        normalized = self._normalize_text(text)
        if not normalized:
            return 0.0
        text_emb = model.encode(normalized, convert_to_tensor=True)
        return self._sim(keyword_emb, text_emb)

    def _lexical_overlap_score(self, keyword_tokens, text):
        if not keyword_tokens:
            return 0.0

        expanded_keyword_tokens = self._expand_tokens(keyword_tokens)
        text_tokens = self._expand_tokens(self._tokenize(text))

        if not text_tokens:
            return 0.0

        overlap = len(expanded_keyword_tokens.intersection(text_tokens))
        return self._clamp_score(overlap / max(1, len(expanded_keyword_tokens)))

    def _workload_penalty(self, active_project_count):
        if active_project_count <= 0:
            return 0.0
        return min(self.MAX_WORKLOAD_PENALTY, active_project_count * 0.03)

    def _score_project_titles(self, keyword_emb, project_titles, model):
        clean_titles = [self._normalize_text(title) for title in project_titles if self._normalize_text(title)]
        if not clean_titles:
            return 0.0, {}

        unique_titles = list(dict.fromkeys(clean_titles))[:self.MAX_PROJECT_TITLES_FOR_SCORING]
        title_embs = model.encode(unique_titles, convert_to_tensor=True)
        sim_vector = util.cos_sim(keyword_emb, title_embs)[0].tolist()
        clamped_sims = [self._clamp_score(score) for score in sim_vector]
        per_title_map = {title: score for title, score in zip(unique_titles, clamped_sims)}

        top_scores = sorted(clamped_sims, reverse=True)[:self.TOP_PROJECT_TITLES_FOR_SCORE]
        if not top_scores:
            return 0.0, per_title_map

        weights = self._softmax(top_scores)
        aggregate = float(np.sum(np.array(top_scores) * weights))
        return self._clamp_score(aggregate), per_title_map

    def generate_team(self, *, keywords, candidates, include_in_progress=False, num_participants=5):
        keyword_text = self._normalize_text(keywords)
        if not keyword_text:
            return []

        model = self._load_model()
        expanded_keyword_text, keyword_tokens = self._expand_keywords(keyword_text)
        keyword_emb = model.encode(expanded_keyword_text, convert_to_tensor=True)

        results = []
        fallback_candidates = []

        for candidate in candidates:
            user_projects = list(candidate.get('projects', []))
            completed_projects = [p for p in user_projects if p.get('status') == 'COMPLETED']
            active_projects = [p for p in user_projects if p.get('status') in ('IN_PROGRESS', 'NOT_STARTED')]

            if not include_in_progress:
                if active_projects:
                    continue
                scoring_projects = completed_projects
            else:
                scoring_projects = user_projects

            if len(scoring_projects) == 0:
                continue

            degree_text = (candidate.get('degree') or '').strip()
            expertise_text = (candidate.get('expertise') or '').strip()

            completed_project_titles = [p['title'].strip() for p in completed_projects if p.get('title')]
            active_project_titles = [p['title'].strip() for p in active_projects if p.get('title')]

            if include_in_progress:
                project_titles = completed_project_titles + active_project_titles
            else:
                project_titles = completed_project_titles

            degree_score = self._semantic_score(keyword_emb, degree_text, model)
            expertise_score = self._semantic_score(keyword_emb, expertise_text, model)

            completed_project_score, completed_title_semantic_map = self._score_project_titles(
                keyword_emb,
                completed_project_titles,
                model,
            )

            active_project_score = 0.0
            active_title_semantic_map = {}
            if include_in_progress and active_project_titles:
                active_project_score, active_title_semantic_map = self._score_project_titles(
                    keyword_emb,
                    active_project_titles,
                    model,
                )

            if completed_project_titles:
                project_score = completed_project_score
                if include_in_progress and active_project_titles:
                    project_score = self._clamp_score((completed_project_score * 0.85) + (active_project_score * 0.15))
            else:
                project_score = self._clamp_score(active_project_score * 0.75) if include_in_progress else 0.0

            title_semantic_map = {}
            title_semantic_map.update(active_title_semantic_map)
            title_semantic_map.update(completed_title_semantic_map)

            lexical_context = ' '.join([degree_text, expertise_text] + project_titles)
            lexical_score = self._lexical_overlap_score(keyword_tokens, lexical_context)

            semantic_peak = max(degree_score, expertise_score, project_score)
            profile_score = max(degree_score, expertise_score)
            degree_signal = degree_score >= self.DEGREE_MIN_SCORE
            expertise_signal = expertise_score >= self.EXPERTISE_MIN_SCORE
            project_signal = project_score >= self.PROJECT_MIN_SCORE
            has_lexical_signal = lexical_score >= self.MIN_LEXICAL_MATCH_SCORE

            if not (degree_signal or expertise_signal or project_signal):
                continue

            if profile_score < self.PROFILE_MIN_SCORE and not project_signal:
                continue

            if semantic_peak < self.FALLBACK_SEMANTIC_SCORE and not has_lexical_signal:
                continue

            semantic_blend = (
                degree_score * self.DEGREE_WEIGHT +
                expertise_score * self.EXPERTISE_WEIGHT +
                project_score * self.PROJECT_WEIGHT
            )
            final_score = semantic_blend + (lexical_score * self.LEXICAL_WEIGHT)

            if include_in_progress:
                final_score -= self._workload_penalty(len(active_projects))

            final_score = self._clamp_score(final_score)
            if final_score < self.FALLBACK_MIN_FINAL_SCORE:
                continue

            project_details = []
            for project in scoring_projects:
                title = project.get('title')
                if not title:
                    continue
                normalized_title = self._normalize_text(title)
                p_semantic = title_semantic_map.get(normalized_title)
                if p_semantic is None:
                    p_semantic = self._semantic_score(keyword_emb, normalized_title, model)
                p_lexical = self._lexical_overlap_score(keyword_tokens, normalized_title)
                p_score = self._clamp_score(p_semantic + (p_lexical * 0.15))
                project_details.append({
                    'id': project.get('id'),
                    'title': title,
                    'status': project.get('status'),
                    'start_date': project.get('start_date'),
                    'relevance_score': p_score,
                    'is_relevant': (p_score >= max(self.MIN_SEMANTIC_SCORE, 0.25)) or (p_lexical > 0),
                })

            project_details.sort(key=lambda item: item['relevance_score'], reverse=True)

            candidate_payload = {
                'id': candidate.get('id'),
                'name': candidate.get('name'),
                'user': None,
                'degree': candidate.get('degree'),
                'expertise': candidate.get('expertise'),
                'project_titles': project_titles,
                'degree_score': degree_score,
                'expertise_score': expertise_score,
                'project_title_score': project_score,
                'final_score': final_score,
                'campus': candidate.get('campus'),
                'college': candidate.get('college'),
                'total_projects': candidate.get('total_projects', len(user_projects)),
                'ongoing_projects': candidate.get('ongoing_projects', len(active_projects)),
                'projects': project_details,
                '_debug_weighted_scores': {
                    'degree_score': degree_score,
                    'expertise_score': expertise_score,
                    'project_title_score': project_score,
                    'lexical_score': lexical_score,
                    'semantic_blend': semantic_blend,
                    'final_score': final_score,
                    'weights': {
                        'degree': self.DEGREE_WEIGHT,
                        'expertise': self.EXPERTISE_WEIGHT,
                        'project': self.PROJECT_WEIGHT,
                        'lexical': self.LEXICAL_WEIGHT,
                    },
                },
            }

            is_strict_match = (
                final_score >= self.MIN_FINAL_SCORE and (
                    project_score >= self.MIN_SEMANTIC_SCORE
                    or expertise_score >= self.MIN_ANCHOR_SCORE
                    or degree_score >= self.MIN_ANCHOR_SCORE
                )
            )

            if is_strict_match and final_score >= self.MIN_FINAL_SCORE:
                results.append(candidate_payload)
            else:
                fallback_candidates.append(candidate_payload)

        ranking_key = lambda item: (item['final_score'], item['project_title_score'], item['expertise_score'], item['degree_score'])

        results.sort(key=ranking_key, reverse=True)
        if results:
            return results[:num_participants]

        fallback_candidates.sort(key=ranking_key, reverse=True)
        return fallback_candidates[:num_participants]


app = FastAPI(title='WBPMISUESO Team Generation Space', version='1.0.0')
_scorer = TeamScorer()


def _verify_secret(authorization: str | None) -> None:
    expected_secret = os.environ.get('HF_SPACE_SHARED_SECRET', '').strip()
    if not expected_secret:
        return

    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Missing authorization token.')

    provided_secret = authorization.removeprefix('Bearer ').strip()
    if provided_secret != expected_secret:
        raise HTTPException(status_code=403, detail='Invalid authorization token.')


@app.get('/health')
def health_check():
    return {'status': 'ok'}


@app.post('/generate-team')
def generate_team(payload: GenerateTeamRequest, authorization: str | None = Header(default=None)):
    _verify_secret(authorization)

    try:
        team_members = _scorer.generate_team(
            keywords=payload.keywords,
            candidates=payload.candidates,
            include_in_progress=payload.include_in_progress,
            num_participants=payload.num_participants,
        )
        return {
            'success': True,
            'team_members': team_members,
            'count': len(team_members),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc