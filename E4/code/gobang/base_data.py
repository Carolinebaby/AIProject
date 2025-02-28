ROWS = 15
SIDE = 30
EMPTY = -1
BLACK = 1
WHITE = 0

MAX_DEPTH = 4  # 搜索的深度
DIRE = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 方向

# 静态表值
STATIC_TABLE = [[59, 298, 1164, 1200, 200000, 1000000], [20, 178, 1151, 1240, 20000, 2000000]]

PATTERN_TYPE_COUNT = 13
PATTERN_SCORES = [5000000, 4000000, 500, 400, 600, 550, 300, 200, 150, 70, 80, 20, 5]  # 不同棋型的分数
#                  成5      活4       冲4  冲4   活3  活3  眠3  眠3  眠3  活2  活2  眠2 活1
# 不同的棋型
CHENG_5_PATTERN = [[(WHITE, WHITE, WHITE, WHITE, WHITE)], [(BLACK, BLACK, BLACK, BLACK, BLACK)]]
HUO_4_PATTERN = [[(EMPTY, WHITE, WHITE, WHITE, WHITE, EMPTY)], [(EMPTY, BLACK, BLACK, BLACK, BLACK, EMPTY)]]
CHONG_4_PATTERNS1 = [
    [
        (BLACK, WHITE, WHITE, WHITE, WHITE, EMPTY),
        (EMPTY, WHITE, WHITE, WHITE, WHITE, BLACK),
    ],
    [
        (WHITE, BLACK, BLACK, BLACK, BLACK, EMPTY),
        (EMPTY, BLACK, BLACK, BLACK, BLACK, WHITE),
    ]
]
CHONG_4_PATTERNS2 = [
    [
        (WHITE, WHITE, WHITE, EMPTY, WHITE),
        (WHITE, WHITE, EMPTY, WHITE, WHITE),
        (WHITE, EMPTY, WHITE, WHITE, WHITE)
    ],
    [
        (BLACK, BLACK, BLACK, EMPTY, BLACK),                # 111*1
        (BLACK, BLACK, EMPTY, BLACK, BLACK),                # 11*11
        (BLACK, EMPTY, BLACK, BLACK, BLACK)                 # 1*111
    ]
]

LIAN_HUO_3_PATTERN = [
    [
        (EMPTY, WHITE, WHITE, WHITE, EMPTY)
    ],
    [
        (EMPTY, BLACK, BLACK, BLACK, EMPTY)
    ]
]

TIAO_HUO_3_PATTERNS = [
    [
        (EMPTY, WHITE, WHITE, EMPTY, WHITE, EMPTY),
        (EMPTY, WHITE, EMPTY, WHITE, WHITE, EMPTY)
    ],
    [
        (EMPTY, BLACK, BLACK, EMPTY, BLACK, EMPTY),
        (EMPTY, BLACK, EMPTY, BLACK, BLACK, EMPTY)
    ]
]

MIAN_3_PATTERNS1 = [
    [
        (BLACK, WHITE, WHITE, WHITE, EMPTY, EMPTY),
        (EMPTY, EMPTY, WHITE, WHITE, WHITE, BLACK),
    ],
    [
        (WHITE, BLACK, BLACK, BLACK, EMPTY, EMPTY),  # 0111**
        (EMPTY, EMPTY, BLACK, BLACK, BLACK, WHITE),  # **1110
    ]
]
MIAN_3_PATTERNS2 = [
    [
        (BLACK, WHITE, WHITE, EMPTY, WHITE, EMPTY),  # 100*0*
        (BLACK, WHITE, EMPTY, WHITE, WHITE, EMPTY),  # 10*00*
        (EMPTY, WHITE, EMPTY, WHITE, WHITE, BLACK),  # *0*001
        (EMPTY, WHITE, WHITE, EMPTY, WHITE, BLACK),  # *00*01
        (WHITE, EMPTY, WHITE, EMPTY, WHITE)              # 0*0*0
    ],
    [
        (WHITE, BLACK, BLACK, EMPTY, BLACK, EMPTY),  # 011*1*
        (WHITE, BLACK, EMPTY, BLACK, BLACK, EMPTY),  # 01*11*
        (EMPTY, BLACK, EMPTY, BLACK, BLACK, WHITE),  # *1*110
        (EMPTY, BLACK, BLACK, EMPTY, BLACK, WHITE),  # *11*10
        (BLACK, EMPTY, BLACK, EMPTY, BLACK)              # 1*1*1
    ]
]
MIAN_3_PATTERNS3 = [
    [
        (BLACK, WHITE, EMPTY, EMPTY, WHITE, WHITE),  # 10**00
        (BLACK, WHITE, WHITE, EMPTY, EMPTY, WHITE),  # 100**0
        (WHITE, WHITE, EMPTY, EMPTY, WHITE, BLACK),  # 00**01
        (WHITE, EMPTY, EMPTY, WHITE, WHITE, BLACK),  # 0**001
        (WHITE, WHITE, EMPTY, EMPTY, WHITE),             # 00**0
        (WHITE, EMPTY, EMPTY, WHITE, WHITE),             # 0**00
    ],
    [
        (WHITE, BLACK, EMPTY, EMPTY, BLACK, BLACK),  # 01**11
        (WHITE, BLACK, BLACK, EMPTY, EMPTY, BLACK),  # 011**1
        (BLACK, BLACK, EMPTY, EMPTY, BLACK, WHITE),  # 11**10
        (BLACK, EMPTY, EMPTY, BLACK, BLACK, WHITE),  # 1**110
        (BLACK, BLACK, EMPTY, EMPTY, BLACK),             # 11**1
        (BLACK, EMPTY, EMPTY, BLACK, BLACK),             # 1**11
    ]
]

HUO_2_PATTERNS1 = [
    [
        (EMPTY, WHITE, WHITE, EMPTY)
    ],
    [
        (EMPTY, BLACK, BLACK, EMPTY)
    ]
]
HUO_2_PATTERNS2 = [
    [
        (EMPTY, WHITE, EMPTY, WHITE, EMPTY),
        (EMPTY, WHITE, EMPTY, EMPTY, WHITE, EMPTY)
    ],
    [
        (EMPTY, BLACK, EMPTY, BLACK, EMPTY),
        (EMPTY, BLACK, EMPTY, EMPTY, BLACK, EMPTY)
    ]
]
MIAN_2_PATTERNS = [
    [
        (EMPTY, EMPTY, EMPTY, WHITE, WHITE, BLACK),
        (BLACK, WHITE, WHITE, EMPTY, EMPTY, EMPTY),
        (EMPTY, EMPTY, WHITE, EMPTY, WHITE, BLACK),
        (BLACK, WHITE, EMPTY, WHITE, EMPTY, EMPTY),
        (EMPTY, WHITE, EMPTY, EMPTY, WHITE, BLACK),
        (BLACK, WHITE, EMPTY, EMPTY, WHITE, EMPTY)
    ],
    [
        (EMPTY, EMPTY, EMPTY, BLACK, BLACK, WHITE),
        (WHITE, BLACK, BLACK, EMPTY, EMPTY, EMPTY),
        (EMPTY, EMPTY, BLACK, EMPTY, BLACK, WHITE),
        (WHITE, BLACK, EMPTY, BLACK, EMPTY, EMPTY),
        (EMPTY, BLACK, EMPTY, EMPTY, BLACK, WHITE),
        (WHITE, BLACK, EMPTY, EMPTY, BLACK, EMPTY),
        (BLACK, EMPTY, EMPTY, EMPTY, BLACK)
    ]
]
HUO_1_PATTERNS = [
    [(EMPTY, EMPTY, WHITE, EMPTY, EMPTY)],
    [(EMPTY, EMPTY, BLACK, EMPTY, EMPTY)]
]
PATTERNS = [
    [
        CHENG_5_PATTERN[0], HUO_4_PATTERN[0], CHONG_4_PATTERNS1[0], CHONG_4_PATTERNS2[0],
        LIAN_HUO_3_PATTERN[0], TIAO_HUO_3_PATTERNS[0], MIAN_3_PATTERNS1[0], MIAN_3_PATTERNS2[0], MIAN_3_PATTERNS3[0],
        HUO_2_PATTERNS1[0], HUO_2_PATTERNS2[0], MIAN_2_PATTERNS[0], HUO_1_PATTERNS[0]
    ],
    [
        CHENG_5_PATTERN[1], HUO_4_PATTERN[1], CHONG_4_PATTERNS1[1], CHONG_4_PATTERNS2[1],
         LIAN_HUO_3_PATTERN[1], TIAO_HUO_3_PATTERNS[1], MIAN_3_PATTERNS1[1], MIAN_3_PATTERNS2[1], MIAN_3_PATTERNS3[1],
         HUO_2_PATTERNS1[1], HUO_2_PATTERNS2[1], MIAN_2_PATTERNS[1], HUO_1_PATTERNS[1]
    ]
]

# CHENG_5_PATTERN = [(color, color, color, color, color)]
# HUO_4_PATTERN = [(EMPTY, color, color, color, color, EMPTY)]
# CHONG_4_PATTERNS1 = [
#     (not color, color, color, color, color, EMPTY),  # 01111*
#     (EMPTY, color, color, color, color, not color),  # *11110
# ]
# CHONG_4_PATTERNS2 = [
#     (color, color, color, EMPTY, color),                # 111*1
#     (color, color, EMPTY, color, color),                # 11*11
#     (color, EMPTY, color, color, color)                 # 1*111
# ]
# LIAN_HUO_3_PATTERN = [
#     (EMPTY, color, color, color, EMPTY)
# ]
# TIAO_HUO_3_PATTERNS = [(EMPTY, color, color, EMPTY, color, EMPTY),
#                        (EMPTY, color, EMPTY, color, color, EMPTY)]
# # HUO_3_PATTERNS = [(EMPTY, color, color, color, EMPTY),
# #                   (EMPTY, color, color, EMPTY, color, EMPTY),
# #                   (EMPTY, color, EMPTY, color, color, EMPTY)]
# MIAN_3_PATTERNS1 = [
#     (not color, color, color, color, EMPTY, EMPTY),  # 0111**
#     (EMPTY, EMPTY, color, color, color, not color),  # **1110
# ]
# MIAN_3_PATTERNS2 = [
#     (not color, color, color, EMPTY, color, EMPTY),  # 011*1*
#     (not color, color, EMPTY, color, color, EMPTY),  # 01*11*
#     (EMPTY, color, EMPTY, color, color, not color),  # *1*110
#     (EMPTY, color, color, EMPTY, color, not color),  # *11*10
#     (color, EMPTY, color, EMPTY, color)              # 1*1*1
# ]
# MIAN_3_PATTERNS3 = [
#     (not color, color, EMPTY, EMPTY, color, color),  # 01**11
#     (not color, color, color, EMPTY, EMPTY, color),  # 011**1
#     (color, color, EMPTY, EMPTY, color, not color),  # 11**10
#     (color, EMPTY, EMPTY, color, color, not color),  # 1**110
#     (color, color, EMPTY, EMPTY, color),             # 11**1
#     (color, EMPTY, EMPTY, color, color),             # 1**11
#
# ]
# HUO_2_PATTERNS1 = [
#     (EMPTY, color, color, EMPTY)
# ]
# HUO_2_PATTERNS2 = [
#     (EMPTY, color, EMPTY, color, EMPTY),
#     (EMPTY, color, EMPTY, EMPTY, color, EMPTY)
# ]
# MIAN_2_PATTERNS = [(EMPTY, EMPTY, EMPTY, color, color, not color),
#                    (not color, color, color, EMPTY, EMPTY, EMPTY),
#                    (EMPTY, EMPTY, color, EMPTY, color, not color),
#                    (not color, color, EMPTY, color, EMPTY, EMPTY),
#                    (EMPTY, color, EMPTY, EMPTY, color, not color),
#                    (not color, color, EMPTY, EMPTY, color, EMPTY),
#                    (color, EMPTY, EMPTY, EMPTY, color)]
