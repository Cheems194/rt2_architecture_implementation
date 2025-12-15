class MixedBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        robot_len: int,
        vlm_len: int,
        batch_size: int,
        robot_ratio: float = 0.7
    ):
        self.robot_len = robot_len
        self.vlm_len = vlm_len
        self.batch_size = batch_size

        self.robot_bs = int(batch_size * robot_ratio)
        self.vlm_bs = batch_size - self.robot_bs

    def __iter__(self):
        robot_indices = list(range(self.robot_len))
        vlm_indices = list(range(self.vlm_len))

        random.shuffle(robot_indices)
        random.shuffle(vlm_indices)

        r_ptr, v_ptr = 0, 0

        while r_ptr < self.robot_len and v_ptr < self.vlm_len:
            batch = []

            batch.extend(robot_indices[r_ptr:r_ptr+self.robot_bs])
            batch.extend(
                [i + self.robot_len for i in vlm_indices[v_ptr:v_ptr+self.vlm_bs]]
            )

            random.shuffle(batch)

            yield batch

            r_ptr += self.robot_bs
            v_ptr += self.vlm_bs

    def __len__(self):
        return min(
            self.robot_len // self.robot_bs,
            self.vlm_len // self.vlm_bs
        )
