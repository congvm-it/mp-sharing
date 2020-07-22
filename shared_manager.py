import multiprocessing
from multiprocessing.managers import BaseManager


class SharedManager():
    def __init__(self):
        self.sharing_manager = BaseManager()
        self.register_cls = []

    def register(self, cls, *args, **kwargs):
        cls_name = cls.__name__
        self.sharing_manager.register(cls_name, cls)
        self.register_cls.append([cls_name, args, kwargs])

    def get_shared_instances(self):
        self.sharing_manager.start()
        shared_dict = {}
        for cls_name, args, kwargs in self.register_cls:
            cls = getattr(self.sharing_manager, cls_name, None)
            if cls is not None:
                inst = cls(*args, **kwargs)
                shared_dict[cls_name] = inst
            else:
                shared_dict[cls_name] = None
        return shared_dict
