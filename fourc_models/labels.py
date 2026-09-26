# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Label vocabularies — must match the Classroom Sentiment Labeling Guide (v1.2)
and the CSV output format it defines. Change both together."""

GUIDE_VERSION = "1.2"

SENTIMENT = ["positive", "neutral", "negative"]
MOVE = ["question", "reasoning", "new_idea", "builds_on", "coordinates", "stuck", "off_task", "other"]
CT_SKILL = [
    "none",
    "interpretation",
    "analysis",
    "evaluation",
    "inference",
    "explanation",
    "self_regulation",
]
ARGUMENT = ["none", "claim", "evidence", "challenge", "revision"]
IDEA_LINK = ["new", "repeat", "develops", "combines"]
PAIR_LABEL = ["different", "related", "same"]
GRADE_BANDS = ["K-2", "3-5", "6-8", "9-12"]

#: Heads of the multi-task utterance model, in a fixed order.
UTTERANCE_HEADS = {
    "sentiment": SENTIMENT,
    "move": MOVE,
    "ct_skill": CT_SKILL,
    "argument": ARGUMENT,
}

#: Rubric levels for segment ratings; "NE" = not enough evidence (excluded from training).
LEVELS = [1, 2, 3, 4]
NOT_ENOUGH_EVIDENCE = "NE"

#: Display mapping used by the app (rubric level -> 1-10 scale).
LEVEL_TO_DISPLAY = {1: 2.5, 2: 5.0, 3: 7.5, 4: 10.0}
