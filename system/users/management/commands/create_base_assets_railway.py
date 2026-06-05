from decimal import Decimal
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from faker import Faker
from django.conf import settings
import os
import random

from system.users.models import College, Campus
from shared.projects.models import SustainableDevelopmentGoal, ProjectType
from shared.downloadables.models import Downloadable


COLLEGES = [
    "College of Arts and Humanities",
    "College of Business and Accountancy",
    "College of Criminal Justice Education",
    "College of Engineering",
    "College of Architecture",
    "College of Hospitality Management and Tourism",
    "College of Nursing and Health Sciences",
    "College of Sciences",
    "College of Teacher Education",
    "PSU PCAT Cuyo",
    "PSU Araceli",
    "PSU Balabac",
    "PSU Bataraza",
    "PSU Brooke’s Point",
    "PSU Coron",
    "PSU Dumaran",
    "PSU El Nido",
    "PSU Linapacan",
    "PSU Narra",
    "PSU Quezon",
    "PSU Rizal",
    "PSU Roxas",
    "PSU San Rafael",
    "PSU San Vicente",
    "PSU Sofronio Española",
    "PSU Taytay",
    "Graduate School",
    "School of Law",
    "School of Medicine",
]

CAMPUSES = [
    "Tinuigiban", "Rizal", "Narra", "Quezon", "Araceli", "Brooke's Point",
    "San Vicente", "Cuyo", "Coron", "Balabac", "Roxas", "Taytay",
    "El Nido", "Linapacan", "San Rafael", "Sofronio Española",
    "Dumaran", "Bataraza",
]

SDG_DATA = [
    {'goal_number': 1, 'name': 'No Poverty'},
    {'goal_number': 2, 'name': 'Zero Hunger'},
    {'goal_number': 3, 'name': 'Good Health and Well-being'},
    {'goal_number': 4, 'name': 'Quality Education'},
    {'goal_number': 5, 'name': 'Gender Equality'},
    {'goal_number': 6, 'name': 'Clean Water and Sanitation'},
    {'goal_number': 7, 'name': 'Affordable and Clean Energy'},
    {'goal_number': 8, 'name': 'Decent Work and Economic Growth'},
    {'goal_number': 9, 'name': 'Industry, Innovation and Infrastructure'},
    {'goal_number': 10, 'name': 'Reduced Inequality'},
    {'goal_number': 11, 'name': 'Sustainable Cities and Communities'},
    {'goal_number': 12, 'name': 'Responsible Consumption and Production'},
    {'goal_number': 13, 'name': 'Climate Action'},
    {'goal_number': 14, 'name': 'Life Below Water'},
    {'goal_number': 15, 'name': 'Life on Land'},
    {'goal_number': 16, 'name': 'Peace, Justice and Strong Institutions'},
    {'goal_number': 17, 'name': 'Partnerships for the Goals'},
]


# Faker seed keeps demo data predictable across runs for recordings.
Faker.seed(2026)
random.seed(2026)
fake = Faker("en_PH")


# Keep quick-login credentials the same, but make on-screen names look realistic.
QUICK_LOGIN_PROFILES = {
    'VP': {'given_name': 'Jose', 'middle_initial': 'P', 'last_name': 'Rizal', 'sex': 'MALE'},
    'DIRECTOR': {'given_name': 'Jose', 'middle_initial': 'P', 'last_name': 'Rizal', 'sex': 'MALE'},
    'UESO': {'given_name': 'Jose', 'middle_initial': 'P', 'last_name': 'Rizal', 'sex': 'MALE'},
    'COORDINATOR': {'given_name': 'Jose', 'middle_initial': 'P', 'last_name': 'Rizal', 'sex': 'MALE'},
    'DEAN': {'given_name': 'Jose', 'middle_initial': 'P', 'last_name': 'Rizal', 'sex': 'MALE'},
    'PROGRAM_HEAD': {'given_name': 'Jose', 'middle_initial': 'P', 'last_name': 'Rizal', 'sex': 'MALE'},
    'FACULTY': {'given_name': 'Jose', 'middle_initial': 'P', 'last_name': 'Rizal', 'sex': 'MALE'},
    'IMPLEMENTER': {'given_name': 'Jose', 'middle_initial': 'P', 'last_name': 'Rizal', 'sex': 'MALE'},
    'CLIENT': {'given_name': 'Jose', 'middle_initial': 'P', 'last_name': 'Rizal', 'sex': 'MALE'},
}

class Command(BaseCommand):
    help = "Populate test users, colleges, campuses, SDGs, project types, agendas, client requests, and downloadables (idempotent)."

    def handle(self, *args, **kwargs):
        User = get_user_model()
        from django.utils import timezone

        # --- CAMPUSES ---
        if Campus.objects.exists():
            self.stdout.write(self.style.WARNING("Campuses already populated — skipping.\n"))
        else:
            self.stdout.write("Populating campuses...")
            campus_objs = {}
            for name in CAMPUSES:
                obj, _ = Campus.objects.get_or_create(name=name)
                campus_objs[name] = obj
            self.stdout.write(self.style.SUCCESS(f"Created {len(campus_objs)} campuses.\n"))

        campus_objs = {c.name: c for c in Campus.objects.all()}

        # --- COLLEGES ---
        # Robust college logo copy logic: check staticfiles first, then static
        staticfiles_logo_dir = os.path.join(settings.BASE_DIR, 'staticfiles', 'colleges', 'logos')
        static_logo_dir = os.path.join(settings.BASE_DIR, 'static', 'colleges', 'logos')
        if os.path.exists(staticfiles_logo_dir) and os.listdir(staticfiles_logo_dir):
            logo_dir = staticfiles_logo_dir
        else:
            logo_dir = static_logo_dir
        tinuigiban_campus = campus_objs.get("Tinuigiban")
        media_logo_dir = os.path.join(settings.MEDIA_ROOT, 'colleges', 'logos')
        os.makedirs(media_logo_dir, exist_ok=True)
        for name in COLLEGES:
            for ext in [".png", ".jpg", ".jpeg", ".svg"]:
                src_logo_path = os.path.join(logo_dir, f"{name}{ext}")
                dest_logo_path = os.path.join(media_logo_dir, f"{name}{ext}")
                if os.path.exists(src_logo_path):
                    import shutil
                    shutil.copy2(src_logo_path, dest_logo_path)
        # Default logo
        default_logo_src = os.path.join(logo_dir, "Default.png")
        default_logo_dest = os.path.join(media_logo_dir, "Default.png")
        if os.path.exists(default_logo_src):
            import shutil
            shutil.copy2(default_logo_src, default_logo_dest)

        college_objs_list = []
        for name in COLLEGES:
            obj, _ = College.objects.get_or_create(name=name)
            campus_match = next((campus for cname, campus in campus_objs.items() if cname in name), None)
            obj.campus = campus_match or tinuigiban_campus
            obj.save(update_fields=["campus"])
            college_objs_list.append(obj)

            # Assign logo if exists in static
            logo_assigned = False
            for ext in [".png", ".jpg", ".jpeg", ".svg"]:
                logo_path = os.path.join(logo_dir, f"{name}{ext}")
                if os.path.exists(logo_path):
                    obj.logo = f"colleges/logos/{name}{ext}"
                    obj.save(update_fields=["logo"])
                    logo_assigned = True
                    break
            if not logo_assigned:
                default_logo = os.path.join(logo_dir, "Default.png")
                if os.path.exists(default_logo):
                    obj.logo = "colleges/logos/Default.png"
                    obj.save(update_fields=["logo"])
        self.stdout.write(self.style.SUCCESS(f"Created {len(college_objs_list)} colleges.\n"))

        # --- USERS ---
        self.stdout.write("Checking test users...")
        roles = [choice[0] for choice in User.Role.choices]
        default_password = "test1234"
        college_of_sciences = College.objects.filter(name="College of Sciences").first()
        created_users = 0
        updated_users = 0
        for role in roles:
            email = f"{role.lower()}@example.com"
            profile = QUICK_LOGIN_PROFILES.get(role, {
                'given_name': role.replace('_', ' ').title().split(' ')[0],
                'middle_initial': 'P',
                'last_name': 'Rizal',
                'sex': 'MALE',
            })
            assigned_college = college_of_sciences if role in [
                User.Role.COORDINATOR,
                User.Role.DEAN,
                User.Role.PROGRAM_HEAD,
                User.Role.FACULTY,
            ] else None

            user = User.objects.filter(email=email).first()
            if not user:
                User.objects.create_user(
                    username=role.lower(),
                    email=email,
                    password=default_password,
                    given_name=profile['given_name'],
                    middle_initial=profile['middle_initial'],
                    last_name=profile['last_name'],
                    sex=User.Sex.FEMALE if profile['sex'] == 'FEMALE' else User.Sex.MALE,
                    contact_no="0999999999",
                    college=assigned_college,
                    role=role,
                    is_confirmed=True,
                )
                created_users += 1
                continue

            changed = False
            target_sex = User.Sex.FEMALE if profile['sex'] == 'FEMALE' else User.Sex.MALE
            target_values = {
                'username': role.lower(),
                'given_name': profile['given_name'],
                'middle_initial': profile['middle_initial'],
                'last_name': profile['last_name'],
                'sex': target_sex,
                'contact_no': '0999999999',
                'college': assigned_college,
                'role': role,
                'is_confirmed': True,
            }
            for field_name, value in target_values.items():
                if getattr(user, field_name) != value:
                    setattr(user, field_name, value)
                    changed = True

            # Keep quick-login credentials stable for demo recordings.
            if not user.check_password(default_password):
                user.set_password(default_password)
                changed = True

            if changed:
                user.save()
                updated_users += 1

        if created_users or updated_users:
            self.stdout.write(self.style.SUCCESS(
                f"Quick-login users ready: {created_users} created, {updated_users} updated (credentials preserved).\n"
            ))
        else:
            self.stdout.write(self.style.WARNING("Quick-login users already up to date — skipping.\n"))

        # --- SDGs ---
        if SustainableDevelopmentGoal.objects.exists():
            self.stdout.write(self.style.WARNING("SDGs already populated — skipping.\n"))
        else:
            for sdg in SDG_DATA:
                SustainableDevelopmentGoal.objects.get_or_create(
                    goal_number=sdg["goal_number"],
                    defaults={"name": sdg["name"]}
                )
            self.stdout.write(self.style.SUCCESS(f"Created {len(SDG_DATA)} SDGs.\n"))

        # --- PROJECT TYPES (NEW) ---
        if ProjectType.objects.exists():
            self.stdout.write(self.style.WARNING("Project Types already populated — skipping.\n"))
        else:
            self.stdout.write("Populating Project Types...")
            project_types = ["Needs Based", "Research Based"]
            for pt_name in project_types:
                ProjectType.objects.get_or_create(name=pt_name)
            self.stdout.write(self.style.SUCCESS(f"Created {len(project_types)} project types.\n"))

        # --- AGENDAS ---
        from internal.agenda.models import Agenda
        if Agenda.objects.exists():
            self.stdout.write(self.style.WARNING("Agendas already populated — skipping.\n"))
        else:
            self.stdout.write("Populating agendas with concerned colleges...")

            def get_colleges(names):
                abbr = {
                    'CBA': 'College of Business and Accountancy',
                    'CAH': 'College of Arts and Humanities',
                    'CS': 'College of Sciences',
                    'CHTM': 'College of Hospitality Management and Tourism',
                    'CNHS': 'College of Nursing and Health Sciences',
                    'CEAT': 'College of Engineering',
                    'CCJE': 'College of Criminal Justice Education',
                    'CTE': 'College of Teacher Education',
                    'Graduate School': 'Graduate School',
                    'School of Medicine': 'School of Medicine',
                    'School of Law': 'School of Law',
                }
                results = []
                for n in names:
                    if n == "External Campus":
                        results.extend([c for c in College.objects.all() if c.campus and c.campus.name != "Tinuigiban"])
                    else:
                        full = abbr.get(n, n)
                        col = College.objects.filter(name=full).first()
                        if col:
                            results.append(col)
                return list(set(results))

            director = User.objects.filter(role=User.Role.DIRECTOR).first()
            agenda_samples = [
                {
                    'name': 'Economics, Entrepreneurship, and Livelihood Enhancement',
                    'description': 'Programs to enhance livelihood, entrepreneurship, and economic sustainability.',
                    'colleges': ['CBA', 'External Campus', 'Graduate School']
                },
                {
                    'name': 'Environmental IEC and Culture Sensitivity',
                    'description': 'Initiatives promoting environmental awareness, culture sensitivity, and sustainable practices.',
                    'colleges': ['CAH', 'CS']
                },
                {
                    'name': 'Hospitality and Tourism Industry Enhancement',
                    'description': 'Programs supporting tourism development, service quality, and industry innovation.',
                    'colleges': ['CHTM', 'External Campus']
                },
                {
                    'name': 'Agriculture, Environmental Protection, Conservation and Resource Management',
                    'description': 'Projects focused on agricultural sustainability, environmental protection, and resource conservation.',
                    'colleges': ['CS', 'External Campus', 'Graduate School']
                },
                {
                    'name': 'Promotive and Preventive Health, Nutrition and Gender Sensitivity',
                    'description': 'Programs advancing health awareness, nutrition, gender equality, and preventive healthcare.',
                    'colleges': ['CNHS', 'External Campus', 'Graduate School', 'School of Medicine', 'CAH']
                },
                {
                    'name': 'Engineering, Architecture and Appropriate Technology',
                    'description': 'Research and projects applying engineering and architectural innovations for societal benefit.',
                    'colleges': ['CEAT', 'CS', 'External Campus']
                },
                {
                    'name': 'Public Safety and Security, Disaster Risk Management and Governance',
                    'description': 'Initiatives for safety, disaster preparedness, governance, and community resilience.',
                    'colleges': ['CCJE', 'Graduate School', 'CNHS', 'School of Law']
                },
                {
                    'name': 'Literacy and Livelihood Learning Systems',
                    'description': 'Programs fostering literacy, education, and sustainable livelihood development.',
                    'colleges': ['CTE', 'Graduate School', 'CAH', 'External Campus']
                },
                {
                    'name': 'Leadership Enhancement and Governance',
                    'description': 'Training and programs to strengthen leadership, governance, and public administration.',
                    'colleges': ['Graduate School', 'CAH', 'CBA', 'External Campus', 'School of Law']
                },
            ]

            for a in agenda_samples:
                agenda, _ = Agenda.objects.get_or_create(name=a["name"], defaults={"description": a["description"]})
                agenda.concerned_colleges.set(get_colleges(a["colleges"]))
                agenda.created_by = director
                agenda.save()
            self.stdout.write(self.style.SUCCESS(f"Created {len(agenda_samples)} agendas.\n"))

        self.stdout.write(self.style.SUCCESS("✅ Data population completed safely (idempotent).\n"))
        