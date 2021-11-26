# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime: 2020-06-06 11:09:20
@LastEditors: yanxinhao
@Description:
'''

import time
import multiprocessing


class PipelineBase:
    """a module to implement pipeline but don't guarantee the order

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]

    Yields:
        [type]: [description]
    """
    class Worker:
        def __init__(self, worker_idx, work_func):
            self.worker_idx = worker_idx
            self.work_func = work_func
            self.s2c = multiprocessing.Queue()
            self.c2s = multiprocessing.Queue()
            self.p = multiprocessing.Process(
                target=self._process, args=(work_func,))
            self.p.start()

        def _process(self, work_func):
            self.c2s.put("success")
            while True:
                if not self.s2c.empty():
                    data = self.s2c.get()
                    self.c2s.put(work_func(data))
                else:
                    time.sleep(0.001)

    def __init__(self, get_input_func=None, num_workers=4, buffer_size=10):
        self.get_input_func = get_input_func
        self.num_workers = num_workers

        # buffer save the results of workers
        self.buffer_size = buffer_size
        self.buffer = multiprocessing.Queue()

        self.pipeline_module_part = multiprocessing.Process(
            target=self.run, args=(self.buffer, self.buffer_size, self.get_input_func,))
        self.work_func = None
        self.worker_cls = PipelineBase.Worker
        self.on_initialize()

    def on_initialize(self):
        """you should define your own work_func and you can also redefine worker_cls

        Raises:
            ValueError: [description]
        """
        if self.work_func is None:
            raise ValueError("self.work_func should be defined")

    def generate_workers(self):
        for worker_idx in range(self.num_workers):
            yield self.worker_cls(worker_idx, self.work_func)

    def run(self, buffer, buffer_size, get_input_func):
        """there are two types of buffer :buffer_input,buffer(for saving result)

        Args:
            buffer ([type]): [description]
            buffer_size ([type]): [description]
            get_input_func ([type]): [description]

        Returns:
            [type]: [description]
        """
        workers = []
        # set a buffer pool for input data
        buffer_input = []
        max_input_buffer = 2

        def buffer_pool_dispatcher(buffer_pool, s2c):
            if s2c.qsize() == 0 and len(buffer_pool) != 0:
                return buffer_pool.pop(0)
            else:
                return

        # step 1: generate workers
        for worker in self.generate_workers():
            workers.append(worker)

        # step 2: waiting the message of succeed in generating worker
        for worker in workers:
            while True:
                if worker.c2s.get() == "success":
                    break
                time.sleep(0.01)

        # step 3: first running ,initialize running processor
        for worker in workers:
            try:
                data = get_input_func()
                while data is None:
                    data = get_input_func()
                    time.sleep(1)
            except:
                continue
            worker.s2c.put(data)

        print(f"initialized success : {self}")
        # step 4: main loop
        while True:
            # 1.dispatch data
            for worker in workers:
                data = buffer_pool_dispatcher(buffer_input, worker.s2c)
                if data is None:
                    continue
                worker.s2c.put(data)

            # 2.put data into buffer_input and the size can't be over max_input_buffer
            for i in range(len(workers)):
                try:
                    data = get_input_func()
                    if data is None:
                        continue
                    elif len(buffer_input) == max_input_buffer:
                        # print("pipeline input buffer is full ", str(self))
                        buffer_input.pop(0)
                    buffer_input.append(data)
                except:
                    continue

            # 3.save output into buffer(for result)
            for worker in workers:
                while not worker.c2s.empty():
                    if buffer.qsize() == buffer_size:
                        print("pipeline result buffer is full :", str(self))
                        buffer.get()
                    buffer.put(worker.c2s.get())

    def start(self):
        self.pipeline_module_part.start()

    def get_result(self):
        if not self.buffer.empty():
            return self.buffer.get()
        else:
            return None

    def destructor(self):
        self.pipeline_module_part.kill()
