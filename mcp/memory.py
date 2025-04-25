class MemoryStore:
    def __init__(self):
        self.store = {}

    def update(self, context_id, new_data):
        """
        更新指定任务（通过 context_id）对应的内存数据。
        :param context_id: 任务的唯一标识符
        :param new_data: 新的数据（字典格式），将更新任务的内存
        """
        if context_id not in self.store:
            self.store[context_id] = {}
        self.store[context_id].update(new_data)

    def get(self, context_id):
        """
        获取指定任务（通过 context_id）对应的内存数据。
        :param context_id: 任务的唯一标识符
        :return: 任务的内存数据（字典格式）
        """
        return self.store.get(context_id, {})

