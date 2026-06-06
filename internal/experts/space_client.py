from __future__ import annotations

from collections import defaultdict
from typing import Any

import requests
from django.conf import settings
from django.db.models import Q

from shared.projects.models import Project
from system.users.models import User


class HuggingFaceSpaceTeamGenerationError(RuntimeError):
    pass


class HuggingFaceSpaceTeamGenerationClient:
    ELIGIBLE_ROLES = ['FACULTY', 'PROGRAM_HEAD', 'DEAN', 'COORDINATOR', 'DIRECTOR', 'VP']

    def __init__(self, base_url: str | None = None, shared_secret: str | None = None, timeout: int | None = None):
        self.base_url = (base_url or settings.HUGGINGFACE_SPACE_URL or '').rstrip('/')
        self.shared_secret = settings.HUGGINGFACE_SPACE_SHARED_SECRET if shared_secret is None else shared_secret
        self.timeout = settings.HUGGINGFACE_SPACE_TIMEOUT if timeout is None else timeout

    def generate_team(
        self,
        *,
        keywords: str,
        include_in_progress: bool = False,
        campus_filter: str | int | None = None,
        college_filter: str | int | None = None,
        num_participants: int = 5,
    ) -> list[dict[str, Any]]:
        payload = self._build_payload(
            keywords=keywords,
            include_in_progress=include_in_progress,
            campus_filter=campus_filter,
            college_filter=college_filter,
            num_participants=num_participants,
        )
        data = self._post(payload)
        return data.get('team_members', [])

    def _build_payload(
        self,
        *,
        keywords: str,
        include_in_progress: bool,
        campus_filter: str | int | None,
        college_filter: str | int | None,
        num_participants: int,
    ) -> dict[str, Any]:
        keyword_text = (keywords or '').strip()
        if not keyword_text:
            raise ValueError('Keywords are required')

        users = User.objects.filter(
            is_expert=True,
            role__in=self.ELIGIBLE_ROLES,
            is_confirmed=True,
            is_active=True,
        ).select_related('college__campus')

        if campus_filter:
            try:
                users = users.filter(college__campus_id=int(campus_filter))
            except (TypeError, ValueError):
                pass

        if college_filter:
            try:
                users = users.filter(college_id=int(college_filter))
            except (TypeError, ValueError):
                pass

        project_map: dict[int, dict[int, Project]] = defaultdict(dict)
        all_projects = Project.objects.filter(
            Q(project_leader__in=users) | Q(providers__in=users)
        ).distinct().select_related('project_leader').prefetch_related('providers')

        for project in all_projects:
            if project.project_leader_id:
                project_map[project.project_leader_id][project.id] = project

            for provider in project.providers.all():
                project_map[provider.id][project.id] = project

        candidates: list[dict[str, Any]] = []
        for user in users:
            user_projects = list(project_map.get(user.id, {}).values())
            completed_projects = [project for project in user_projects if project.status == 'COMPLETED']
            active_projects = [project for project in user_projects if project.status in ('IN_PROGRESS', 'NOT_STARTED')]

            if not include_in_progress and active_projects:
                continue

            scoring_projects = user_projects if include_in_progress else completed_projects
            if not scoring_projects:
                continue

            candidates.append({
                'id': user.id,
                'name': user.get_full_name(),
                'degree': user.degree,
                'expertise': user.expertise,
                'campus': user.college.campus.name if getattr(user, 'college', None) and getattr(user.college, 'campus', None) and getattr(user.college.campus, 'name', None) else None,
                'college': user.college.name if getattr(user, 'college', None) and getattr(user.college, 'name', None) else None,
                'total_projects': len(user_projects),
                'ongoing_projects': len(active_projects),
                'projects': [
                    {
                        'id': project.id,
                        'title': project.title,
                        'status': project.status,
                        'start_date': project.start_date.strftime('%Y-%m-%d') if project.start_date else None,
                    }
                    for project in scoring_projects
                    if getattr(project, 'title', None)
                ],
            })

        return {
            'keywords': keyword_text,
            'include_in_progress': include_in_progress,
            'campus_filter': campus_filter,
            'college_filter': college_filter,
            'num_participants': num_participants,
            'candidates': candidates,
        }

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.base_url:
            raise HuggingFaceSpaceTeamGenerationError(
                'Hugging Face Space URL is not configured. Set HUGGINGFACE_SPACE_URL first.'
            )

        headers = {'Content-Type': 'application/json'}
        if self.shared_secret:
            headers['Authorization'] = f'Bearer {self.shared_secret}'

        try:
            response = requests.post(
                f'{self.base_url}/generate-team',
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise HuggingFaceSpaceTeamGenerationError(
                f'Failed to reach Hugging Face Space: {exc}'
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise HuggingFaceSpaceTeamGenerationError('Hugging Face Space returned invalid JSON.') from exc

        if not data.get('success'):
            raise HuggingFaceSpaceTeamGenerationError(data.get('error') or 'Hugging Face Space returned an error.')

        return data