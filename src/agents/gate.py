"""Free pre-check before any LLM call: rejects only empty input, single words and clearly off-topic requests.

Everything else goes to the Planner, which decides answerable yes|partly|no.
"""
import re

# What the data lacks; the Planner's gate needs only this, since the schema already shows what it covers.
DATASET_LIMITS = (
    "It covers Sept 2016 to Oct 2018 and has no product names, customer demographics (age, gender), "
    "costs or profit, marketing data, or data after 2018."
)

DATASET_SUMMARY = """\
This assistant answers questions about the Olist Brazilian e-commerce dataset (about 100k orders, Sept 2016 to Oct 2018):
- orders: status, purchase, approval, carrier, delivery and estimated delivery timestamps (late deliveries)
- order items: price and freight per item, product, seller (revenue)
- payments: type, installments, value
- reviews: 1-5 scores and comment text (mostly Portuguese)
- customers and sellers: city, state, zip code prefix (repeat customers via customer_unique_id)
- products: category (Portuguese, with English translation), weight, size, photo count
- geolocation: latitude/longitude per zip code prefix

It has no product names, customer demographics (age, gender), costs or profit, marketing data, or data after 2018."""

OFF_TOPIC = [
    (re.compile(r"\b(weather|forecast|temperature)\b", re.I), "weather"),
    (re.compile(r"\b(poem|poetry|haiku|song lyrics|limerick)\b", re.I), "creative writing"),
    (re.compile(r"\b(write|generate|debug|fix)\b.{0,30}\b(python|javascript|java|code|script|function|program)\b", re.I), "code requests"),
    # "who is the top seller" is a data question, so only names/people after "who is" are rejected.
    (re.compile(r"^\s*who\s+(is|was)\s+(?!the\b|a\b|our\b)", re.I), "questions about people"),
]


def precheck(question: str) -> str | None:
    """Return a short reason to reject the question without calling the LLM, or None to continue."""
    words = question.split()
    if not words:
        return "The question is empty."
    if len(words) == 1:
        return "Please ask a full question, not a single word."
    for pattern, topic in OFF_TOPIC:
        if pattern.search(question):
            return f"This looks like {topic}, which the Olist dataset can't answer."
    return None
