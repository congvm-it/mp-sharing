from multiprocessing.managers import BaseManager


class SharedManager():
    def __init__(self):
        self.sharing_manager = BaseManager()
        self.register_cls = []
        self.shared_dict = {}
        self.index = 0

    def register(self, cls, name, *args, **kwargs):
        self.sharing_manager.register(name, cls)
        self.register_cls.append([name, args, kwargs])
        return self.index

    def __getitem__(self, cls):
        return self.shared_dict[cls]

    def allocate_memory(self):
        self.sharing_manager.start()
        for name, args, kwargs in self.register_cls:
            cls = getattr(self.sharing_manager, name, None)
            if cls is not None:
                inst = cls(*args, **kwargs)
                self.shared_dict[name] = inst
            else:
                self.shared_dict[name] = None
