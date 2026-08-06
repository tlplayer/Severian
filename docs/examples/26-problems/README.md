# Problems

This category is a running corpus of small programming problems solved in
Severian. It serves two purposes: exercise the language with recognizable
algorithms, and expose syntax or library friction that is easy to miss in
feature-focused examples.

Every problem should:

- live in one `.sev` file;
- identify the source set and problem in its leading comments;
- use a direct, readable solution rather than hiding the algorithm in a native
  helper;
- include ordinary tests below the solution, covering the representative cases
  and at least one boundary case;
- compile and execute through the normal documentation-example harness.

The planned order is the official [LeetCode 75][leetcode-75] track, followed by
[Top Interview 150][top-150], followed by increasingly difficult problems
selected for new language pressure. The filename prefix records completion
order. Each source begins with
`Track`, `Problem`, `Difficulty`, and comma-separated `Tags` comments. Tags are
many-to-many: a problem may exercise both `dynamic-programming` and `math`
without duplicating its source file.

[leetcode-75]: https://leetcode.com/studyplan/leetcode-75/
[top-150]: https://leetcode.com/studyplan/top-interview-150/

## Current set

| File | Problem | Language pressure exercised |
| --- | --- | --- |
| `01-kids-with-candies.sev` | LeetCode 75: Kids With the Greatest Number of Candies | List construction, two-pass iteration, booleans. |
| `02-find-highest-altitude.sev` | LeetCode 75: Find the Highest Altitude | Accumulators, negative integers, maximum tracking. |
| `03-find-pivot-index.sev` | LeetCode 75: Find Pivot Index | Multiple passes, indexed iteration, early return. |
| `04-increasing-triplet.sev` | LeetCode 75: Increasing Triplet Subsequence | Sentinel-free state, branching, early return. |
| `05-nth-tribonacci.sev` | LeetCode 75: N-th Tribonacci Number | Constant-space DP, recurrence arithmetic, loop initialization. |
| `06-min-cost-climbing-stairs.sev` | LeetCode 75: Min Cost Climbing Stairs | Constant-space DP, indexed access, loop initialization. |
| `07-house-robber.sev` | LeetCode 75: House Robber | Rolling DP state and local maximum selection. |
| `08-unique-paths.sev` | LeetCode 75: Unique Paths | One-dimensional matrix DP and nested loops. |
| `09-can-place-flowers.sev` | LeetCode 75: Can Place Flowers | Greedy mutation and boundary-safe neighbor checks. |
| `10-product-except-self.sev` | LeetCode 75: Product of Array Except Self | Prefix/suffix products and reverse traversal. |
| `11-move-zeroes.sev` | LeetCode 75: Move Zeroes | In-place mutation and write cursors. |
| `12-container-most-water.sev` | LeetCode 75: Container With Most Water | Converging two-pointer search. |
| `13-maximum-average-subarray.sev` | LeetCode 75: Maximum Average Subarray I | Fixed-width sliding window. |
| `14-max-consecutive-ones.sev` | LeetCode 75: Max Consecutive Ones III | Variable-width sliding window. |
| `15-longest-subarray-after-delete.sev` | LeetCode 75: Longest Subarray After Deleting One | Sliding window with a mandatory deletion. |
| `16-find-array-difference.sev` | LeetCode 75: Find the Difference of Two Arrays | Set conversion, difference, and list materialization. |
| `17-unique-occurrences.sev` | LeetCode 75: Unique Number of Occurrences | Frequency analysis and uniqueness. |
| `18-koko-eating-bananas.sev` | LeetCode 75: Koko Eating Bananas | Binary search over an answer space. |
| `19-successful-pairs.sev` | LeetCode 75: Successful Pairs | Pair counting and threshold arithmetic. |
| `20-domino-tromino-tiling.sev` | LeetCode 75: Domino and Tromino Tiling | Modular recurrence DP. |
| `21-find-peak-element.sev` | LeetCode 75: Find Peak Element | Binary search over local slope. |
| `22-stock-with-transaction-fee.sev` | LeetCode 75: Stock with Transaction Fee | Two-state dynamic programming. |
| `23-counting-bits.sev` | LeetCode 75: Counting Bits | DP over integer halves. |
| `24-single-number.sev` | LeetCode 75: Single Number | Occurrence counting without bitwise syntax. |
| `25-minimum-bit-flips.sev` | LeetCode 75: Minimum Flips for OR | Arithmetic bit decomposition. |
| `26-asteroid-collision.sev` | LeetCode 75: Asteroid Collision | Mutable stack cursor and simulation. |
| `27-daily-temperatures.sev` | LeetCode 75: Daily Temperatures | Forward search and early return. |
| `28-merge-strings-alternately.sev` | LeetCode 75: Merge Strings Alternately | String indexing and concatenation. |
| `29-is-subsequence.sev` | LeetCode 75: Is Subsequence | String indexing and two cursors. |
| `30-maximum-vowels.sev` | LeetCode 75: Maximum Vowels | String sliding window. |
| `31-longest-common-subsequence.sev` | LeetCode 75: Longest Common Subsequence | One-row string DP. |
| `32-edit-distance.sev` | LeetCode 75: Edit Distance | One-row edit DP. |
| `33-gcd-of-strings.sev` | LeetCode 75: GCD of Strings | Euclidean arithmetic and string construction. |
| `34-reverse-vowels.sev` | LeetCode 75: Reverse Vowels | String/character conversion, joining, and two pointers. |
| `35-reverse-words.sev` | LeetCode 75: Reverse Words | Native word splitting, reversal, and joining. |
| `36-string-compression.sev` | LeetCode 75: String Compression | In-place character mutation. |
| `37-max-k-sum-pairs.sev` | LeetCode 75: Max K-Sum Pairs | Pair consumption and logical removal. |
| `38-equal-row-column-pairs.sev` | LeetCode 75: Equal Row and Column Pairs | Nested collection indexing. |
| `39-determine-close-strings.sev` | LeetCode 75: Determine if Two Strings Are Close | Frequency maps, sorted views, and collection equality. |
| `40-removing-stars.sev` | LeetCode 75: Removing Stars From a String | Logical stack truncation and string reconstruction. |
| `41-dota2-senate.sev` | LeetCode 75: Dota2 Senate | Competing logical queues and cyclic scheduling. |
| `42-nearest-exit.sev` | LeetCode 75: Nearest Exit from Entrance in Maze | Matrix mutation and breadth-first search. |
| `43-rotting-oranges.sev` | LeetCode 75: Rotting Oranges | Multi-source breadth-first search. |
| `44-number-of-provinces.sev` | LeetCode 75: Number of Provinces | Recursive DFS and union-find. |
| `45-course-schedule.sev` | Top Interview 150: Course Schedule | Topological sorting with an indegree queue. |
| `46-range-sum-mutable.sev` | Top Interview 150: Range Sum Query - Mutable | Iterative segment tree construction, update, and query. |
| `47-kth-largest-element.sev` | LeetCode 75: Kth Largest Element | Bounded min-heap and concise sorted variants. |
| `48-n-queens-ii.sev` | Hard validation: N-Queens II | Recursive backtracking with arithmetic bitmasks. |
| `49-merge-sort.sev` | Hard validation: Sort an Array | Bottom-up merge sort. |
| `50-polynomial-string-search.sev` | Hard validation: Find First Occurrence | Polynomial rolling hash with collision verification. |
| `51-fenwick-tree.sev` | Hard validation: Range Sum Query - Mutable | Fenwick tree updates and prefix queries. |
| `52-check-straight-line.sev` | Hard validation: Check Straight Line | Cross-product geometry. |
| `53-quickselect.sev` | Hard validation: Kth Largest Element | In-place quickselect partitioning. |
| `54-largest-rectangle-histogram.sev` | Hard validation: Largest Rectangle in Histogram | Monotonic stack with a sentinel iteration. |
| `55-sliding-window-maximum.sev` | Hard validation: Sliding Window Maximum | Logical monotonic deque. |
| `56-validate-binary-search-tree.sev` | Top Interview 150: Validate BST | Recursive bound propagation over a binary tree. |
| `57-predict-the-winner.sev` | Hard validation: Predict the Winner | Memoized minimax and game state. |
| `58-counting-sort.sev` | Hard validation: Sort an Array | Counting and bucket sort over signed values. |
| `59-implement-trie.sev` | LeetCode 75: Implement Trie | Array-backed prefix-tree construction and lookup. |
| `60-suffix-array.sev` | Hard validation: Suffix Array Construction | Lexicographic suffix ordering. |
| `61-car-pooling.sev` | Hard validation: Car Pooling | Sweep-line deltas and prefix accumulation. |
| `62-min-cost-connect-points.sev` | Hard validation: Min Cost to Connect Points | Prim minimum spanning tree over Manhattan distance. |
| `63-closest-subsequence-sum.sev` | Hard validation: Closest Subsequence Sum | Meet-in-the-middle subset enumeration. |
| `64-reservoir-sampling.sev` | Hard validation: Random Pick Index | Seeded reservoir sampling over a stream. |

Progress: **64** total problems, including **46 / 75** from LeetCode 75,
with **200** attached test cases.

## Technique coverage

The gallery now directly exercises array and string processing, hash tables,
math and number theory, dynamic programming and memoization, sorting, greedy
algorithms, DFS and BFS, binary search, bit manipulation and arithmetic
bitmasks, matrices, trees and binary-search trees, prefix sums, two pointers,
heaps, simulation, counting, graphs, stacks and monotonic stacks, sliding
windows and monotonic queues, enumeration, data-structure design,
backtracking, union-find, segment and Fenwick trees, divide-and-conquer,
combinatorics, tries, queues, recursion, geometry, shortest paths,
topological sorting, string matching and rolling hashes, game theory and
minimax, merge/counting/bucket sorting, data streams, suffix arrays,
quickselect, sweep lines, probability, minimum spanning trees,
meet-in-the-middle, and reservoir sampling.

## Syntax observations

Keep observations here even when the current syntax is valid. Repeated friction
is stronger evidence for a language or standard-library change than a single
contrived example.

- Collections expose `minimum()`, `maximum()`, `sum()`, and `sorted()` as
  ownership-preserving native operations. Explicit algorithm variants remain
  in the gallery when the implementation technique itself is under test.
- Index-sensitive algorithms use `indices(values)`, which is readable and keeps
  the collection's shape stable while iterating.
- Returning a list still requires a concrete annotation such as `list[bool]`;
  local empty-list element types are inferred from later appends.
- There are no integer infinity literals, so the increasing-triplet solution
  uses explicit `hasFirst` and `hasSecond` state instead of numeric sentinels.
- Loop initialization reads naturally for a single value (`while index <= n
  with index := 3`), while algorithms needing several loop-local cursors still
  initialize the remaining state immediately above the loop.
- `pop()` and `last()` remove the logical-cursor boilerplate from ordinary stack
  algorithms. Shape-stable iteration still rejects mutation of the collection
  being traversed.
- `characters()`, `words()`, `split()`, and `join()` let string problems state
  their transformation directly while lowering to native loops and buffers.
- `frequencies()`, map `keys()`/`values()`, and set conversion/difference cover
  common counting and membership idioms without hand-written nested scans.
- Queue algorithms similarly keep a logical `head` cursor because removing the
  first list element would otherwise require shifting the remaining values.
- Matrix BFS is readable with parallel row, column, and distance queues, though
  a tuple-valued queue would express the relationship between those values more
  directly.
- A list grown entirely through a mutating helper retains its initial fixed
  shape in semantic analysis, so heap construction seeds the first element
  locally before delegating later insertions.
- The off-by-one range diagnostic treats every `range(0, size(values) + 1)` as
  collection indexing, even when the final iteration is an intentional
  sentinel or the loop only appends to a different collection. Such loops
  currently use explicit `while` conditions or seed-and-append construction.

When a syntax or library improvement lands, update the solution and retain a
short note describing what became simpler.
