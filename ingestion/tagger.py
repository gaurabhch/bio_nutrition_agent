from config import CATEGORY_TO_DOMAIN


def tag_domain(obj):
    obj.domain = CATEGORY_TO_DOMAIN.get(obj.category, 'pcos_general')
    return obj


# def tag_all(question_variants: list, content_chunks: list):
#     return (
#         [tag_domain(q) for q in question_variants],
#         [tag_domain(c) for c in content_chunks]
#     )

def tag_all(question_variants: list, content_chunks: list):
    return question_variants, content_chunks