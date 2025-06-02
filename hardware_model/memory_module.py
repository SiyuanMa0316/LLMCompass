class MemoryModule:
    def __init__(self, memory_capacity):
        self.memory_capacity = memory_capacity
    def info(self):
        return f"Memory capacity: {self.memory_capacity / 2**30} GB"
memory_module_dict = {'A100_80GB': MemoryModule(80e9),'TPUv3': MemoryModule(float('inf')),'MI210': MemoryModule(64e9),'PIMSAB': MemoryModule(80e9),'SIMDRAM': MemoryModule(80e9)}
