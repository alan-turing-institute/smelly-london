import nltk
import math

print("reading Brown corpus...")

brown_words = list(nltk.corpus.brown.words())
import nltk
import math

print("reading Brown corpus...")

brown_words = list(nltk.corpus.brown.words())

print("computing space...")

context_size = 10
space = nltk.ConditionalFreqDist()

for index in range(len(brown_words)):

    # current word
    current = brown_words[ index ]

    # context before the current word: count each item
    # but no preceding context for index 0
    if index > 0:
        # don't start from a cxword_index < 0 in case index < context_size

        for cxword_index in range(max(index - context_size, 0), index):
            cxword = brown_words[ cxword_index ]

            # In a ConditionalFreqDist, if 'current' is not a condition yet,
            # then accessing it creates a new empty FreqDist for 'current'
            # The FreqDist method inc() increments the count for the given item by one.
            #space[ current ].inc(cxword)
            space[current][cxword] += 1

    # context after the current word: count each item
    # but no succeeding context for the last item (index len(brown_words - 1))
    if index < len(brown_words) - 1:

        # don't run until a cxword_index > len(brown_words) in case
        # index + context_size > len(brown_words)
        for cxword_index in range(index + 1, min(index + context_size + 1, len(brown_words))):

            cxword = brown_words[ cxword_index ]

            # In a ConditionalFreqDist, if 'current' is not a condition yet,
            # then accessing it creates a new empty FreqDist for 'current'
            # The FreqDist method inc() increments the count for the given item by one.
            #space[ current ].inc(cxword)
            space[current][cxword] += 1


print("COUNTS FOR 'election':")
#for cxword, count in space[ 'election' ].items()[:50]:
#    print(cxword, ":", count)

print("COUNTS FOR 'water':")
#for cxword, count in space[ 'water' ].items()[:50]:
#    print(cxword, ":", count)


# cosine similarity between word1 and word2:
#
# sum_w space[word1][w] * space[word2][w]
# -----------------------------------------------------------
# sqrt(sum_w space[word1][w]^2) * sqrt(sum_w space[word2][w]^2)
#
def cosine(space, word1, word2):
    denominator = math.sqrt(sum([count*count for count in space[word1].values()])) * math.sqrt(sum([count * count for count in space[word2].values()]))
    numerator = sum([ space[word1][w] * space[word2][w] for w in space[word1].keys() ])
    return float(numerator) / float(denominator)

# some word similarities

print(cosine(space, "fire", "water"))
print(cosine(space, "election", "vote"))
print(cosine(space, "the", "happy"))
print(cosine(space, "good", "evil"))
print(cosine(space, "good", "bad"))
print("computing space...")
