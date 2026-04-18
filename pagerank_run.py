from collections import defaultdict

file_path = r"D:\PGDDE\Mtech\Big data and ML\Assignment 4\Assignment 4- datasets\Q2- webSearch\actions.txt"

"""Reading Grapghs"""
links = defaultdict(list)
pages = set()

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            src, dst = parts[0], parts[1]
            links[src].append(dst)
            pages.add(src)
            pages.add(dst)

"""Initializing Rank"""
N = len(pages)
ranks = {p: 1 / N for p in pages}
d = 0.85
iterations = 10

for _ in range(iterations):
    new_ranks = {p: (1 - d) / N for p in pages}

    for page in pages:
        outs = links.get(page, [])
        if outs:
            share = ranks[page] / len(outs)
            for dest in outs:
                new_ranks[dest] += d * share
        else:
            # dangling node distributes to all
            share = ranks[page] / N
            for dest in pages:
                new_ranks[dest] += d * share

    ranks = new_ranks

"""Top Pages"""
top_pages = sorted(ranks.items(), key=lambda x: x[1], reverse=True)

print("Top 10 Pages by Rank:")
for page, rank in top_pages[:10]:
    print(page, round(rank, 6))