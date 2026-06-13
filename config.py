import os
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv

load_dotenv(override=True)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_FAST = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
BATCH_SIZE = 64

DATABASE_URL = os.environ["DATABASE_URL"]


def _make_async_url(url: str) -> str:
    parsed = urlparse(url)
    clean = parsed._replace(scheme="postgresql+asyncpg", query="")
    return urlunparse(clean)


DATABASE_URL_ASYNC = _make_async_url(DATABASE_URL)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CONVERSATION_TTL = 86_400
MAX_HISTORY_MSGS = 10
NEON_TABLE_NAME = "nutrition_knowledge_base"

KB_PATH = "biocanvas_nutrition_kb.md"
MAX_FIELD_WORDS = 250
TOP_K_RETRIEVAL = 5
TOP_K_CANDIDATES = 20
KEYWORD_ROUTING_THRESHOLD = 2
AUTO_MERGE_RATIO = 2
FINAL_CONTEXT_TOKENS = 2000
DOMAIN_BOOST_EXACT = 0.08
DOMAIN_BOOST_GENERAL = 0.03
CLUSTER_REPEAT_BOOST = 0.02

CATEGORY_TO_DOMAIN = {
    "Overview & Foundations": "nutrition_general",
    "Core Concepts": "nutrition_general",
    "Food Groups": "food_groups",
    "Macronutrients": "macronutrients",
    "Meal Planning": "meal_planning",
    "Behavioral Nutrition": "behavioral_nutrition",
    "Hydration": "hydration",
    "Gut Health": "gut_health",
    "Supplements": "supplements",
    "Weight Management": "weight_management",
    "Dietary Restrictions": "dietary_restrictions",
    "Indian-Specific Guidance": "indian_guidance",
    "Pregnancy Nutrition": "pregnancy_nutrition",
    "Postpartum Nutrition": "postpartum_nutrition",
    "Fertility": "fertility_nutrition",
}

DOMAIN_TO_DB_CATEGORIES: dict[str, list[str]] = {
    "nutrition_general": ["Overview & Foundations", "Core Concepts"],
    "food_groups": ["Food Groups"],
    "macronutrients": ["Macronutrients", "Food Groups"],
    "meal_planning": ["Meal Planning"],
    "behavioral_nutrition": ["Behavioral Nutrition", "Meal Planning"],
    "hy dration": ["Hydration"],
    "gut_health": ["Gut Health"],
    "supplements": ["Supplements"],
    "weight_management": ["Weight Management"],
    "dietary_restrictions": ["Dietary Restrictions"],
    "indian_guidance": ["Indian-Specific Guidance"],
    "pregnancy_nutrition": ["Pregnancy Nutrition"],
    "postpartum_nutrition": ["Postpartum Nutrition"],
    "fertility_nutrition": ["Fertility"],
}

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "nutrition_general": ["nutrition", "diet", "healthy eating", "food", "meal", "balanced diet"],
    "food_groups": ["food groups", "fruits", "vegetables", "grains", "dairy", "protein foods"],
    "macronutrients": ["protein", "carbs", "carbohydrate", "fat", "fiber", "macros"],
    "meal_planning": ["meal plan", "meal timing", "what should i eat", "diet plan", "routine","breakfast lunch dinner","snacks","satiety", "throughout the day"],
    "behavioral_nutrition": ["cravings", "binge", "eating habit", "emotional eating", "mindful eating"],
    "hydration": ["water", "hydration", "electrolyte", "dehydration", "fluids"],
    "gut_health": ["bloating", "digestion", "gut", "constipation", "diarrhea", "ibs", "acidity"],
    "supplements": ["supplement", "vitamin", "mineral", "omega 3", "magnesium", "iron", "b12"],
    "weight_management": ["weight loss", "weight gain", "calorie", "obesity", "fat loss", "portion"],
    "dietary_restrictions": ["lactose", "gluten", "allergy", "vegan", "vegetarian", "restriction"],
    "indian_guidance": ["Indian woman","indian diet", "roti", "rice", "dal", "paneer", "indian food", "desi diet"],
    "pregnancy_nutrition": ["pregnancy", "pregnant", "prenatal", "trimester"],
    "postpartum_nutrition": ["postpartum", "after delivery", "breastfeeding", "lactation"],
    "fertility_nutrition": ["fertility", "conceive", "trying to conceive", "ovulation", "reproductive health"],
}

RESPONSE_MODES = {
    "INFORMATION": "information",
    "EMOTIONAL": "emotional",
    "CLARIFICATION": "clarification",
    "CRISIS": "crisis",
}

EMOTIONAL_KEYWORDS = [
    "scared", "anxious", "worried", "depressed", "crying", "hopeless",
    "overwhelmed", "lost", "don't know what to do", "help me",
    "frustrated", "sad", "upset",
]

VAGUE_PATTERNS = [
    "i don't feel like myself",
    "something is wrong",
    "i feel weird",
    "i don't know",
    "not sure",
]

CRISIS_KEYWORDS = [
    "suicide", "suicidal", "kill myself", "end my life", "want to die",
    "self harm", "self-harm", "cut myself", "hurt myself", "not worth living",
    "abuse", "being abused", "domestic violence", "overdose",
]

FALSE_POSITIVE_GUARD = ["killing it", "dying of laughter", "headache"]

HELPLINE_RESPONSE = (
    "I can hear that you are going through something very difficult. "
    "Please reach out to someone who can help:\n\n"
    "• iCall: 9152987821\n"
    "• Vandrevala Foundation: 1860-2662-345\n\n"
    "You deserve support and you are not alone."
)

CLUSTER_TABLE_START = 0
FIELD_TYPES = [
    "summary",
    "explanation",
    "actionable",
    "symptoms_it_explains",
    "who_it_affects",
    "red_flags",
]

STREAM_WORD_DELAY = 0.03
GEMINI_TIMEOUT = 10.0
PUBMED_TIMEOUT = 5.0
GROQ_TIMEOUT = 10.0
GROQ_TIMEOUT_MESSAGE = "I am having trouble connecting right now. Please try again in a moment."