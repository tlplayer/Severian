from concurrent.futures import ProcessPoolExecutor


COUNT = 65_536
WORKERS = 4


def relu_shard(values):
    return [value if value > 0.0 else 0.0 for value in values]


def relu_backward_shard(pair):
    values, upstream = pair
    return [gradient if value > 0.0 else 0.0 for value, gradient in zip(values, upstream)]


def shards(values):
    return [
        values[len(values) * worker // WORKERS : len(values) * (worker + 1) // WORKERS]
        for worker in range(WORKERS)
    ]


def main():
    values = [float(index - COUNT // 2) for index in range(COUNT)]
    upstream = [1.0] * COUNT
    value_shards = shards(values)
    upstream_shards = shards(upstream)

    with ProcessPoolExecutor(max_workers=WORKERS) as workers:
        activations = sum(workers.map(relu_shard, value_shards), [])
        gradients = sum(
            workers.map(relu_backward_shard, zip(value_shards, upstream_shards)), []
        )

    print(len(activations))
    print(int(sum(activations)))
    print(int(sum(gradients)))


if __name__ == "__main__":
    main()
