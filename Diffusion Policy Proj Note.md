# Diffusion Policy Proj Note

## Base workspace

### 基类 BaseWorkspace

**作用:**

- 管理训练过程中的配置
- 保存和加载检查点(checkpoint)以及快照(snapshot)

**成员变量:**

    include_keys = tuple()
    exclude_keys = tuple()

说明:定义两个类变量，用于指定保存检查点时需要包含或排除的属性键

**方法：**

- `__init__`方法

            def __init__(self, cfg: OmegaConf, output_dir:  Optional[str]=None):
                    self.cfg = cfg
                    self._output_dir = output_dir
                    self._saving_thread = None
  - **作用:定义初始化方法，接收`OmegaConf`类型的`cfg`参数和可选的`output_dir`并作为成员变量保存**

- `output_dir`属性

        @property
        def output_dir(self):
            output_dir = self._output_dir
            if output_dir is None:
                output_dir = HydraConfig.get().runtime.output_dir
            return output_dir

  - 使用`@property`将`output_dir`方法转化为只读属性，允许通过`instance.output_dir`访问
  - **作用:用户可获取该实例的输出目录**

- `run`方法

        def run(self):
            pass

  - **在子类中重写方法具体实现**

- `save_checkpoint`方法

        def save_checkpoint(self, path=None, tag='latest', 
                exclude_keys=None,
                include_keys=None,
                use_thread=True):
            if path is None:
                path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
            else:
                path = pathlib.Path(path)
            if exclude_keys is None:
                exclude_keys = tuple(self.exclude_keys)
            if include_keys is None:
                include_keys = tuple(self.include_keys) + ('_output_dir',)

            path.parent.mkdir(parents=False, exist_ok=True)
            payload = {
                'cfg': self.cfg,
                'state_dicts': dict(),
                'pickles': dict()
            } 

            for key, value in self.__dict__.items():
                if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                    # modules, optimizers and samplers etc
                    if key not in exclude_keys:
                        if use_thread:
                            payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                        else:
                            payload['state_dicts'][key] = value.state_dict()
                elif key in include_keys:
                    payload['pickles'][key] = dill.dumps(value)
            if use_thread:
                self._saving_thread = threading.Thread(
                    target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
                self._saving_thread.start()
            else:
                torch.save(payload, path.open('wb'), pickle_module=dill)
            return str(path.absolute())

  - 参数:
    - `path`: 字符串，可选，指定保存路径
    - `tag` : 标签，用于标识检查点，默认为`latest`
    - `exclude_keys` : 键列表，可选，指定保存时要排除的属性
    - `include_keys` : 键列表，可选，指定保存时要包含的额外属性
    - `use_thread` : 指示是否使用线程进行保存操作，默认`True`

  - 路径处理:

        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
    - 如果未指定路径，则构造默认的保存路径，位于`output_dir/checkpoints/{tag}.ckpt`，即在该路径保存一个**检查点文件(.ckpt)**
      - *在机器学习和深度学习领域，检查点文件用于保存训练过程中的模型状态。这样，如果训练过程被中断，可以从最近的检查点恢复训练，而不是从头开始。检查点文件还常用于保存训练完成的模型，以便后续的模型评估、测试或部署。*

  - 处理`exclude_keys`和`include_keys`

        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)
        
    - 如果未指定这两个变量，则使用类变量，`include_keys`中还应该包含`_output_dir`属性

  - 创建检查点目录

        path.parent.mkdir(parents=False, exist_ok=True)
  
  - 初始化`payload`

        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        }
    - 创建一个字典，用于保存配置、状态字典、和序列化对象
  - 遍历实例属性并收集需要保存的数据:

        for key, value in self.__dict__.items():
        if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
            # modules, optimizers and samplers etc
            if key not in exclude_keys:
                if use_thread:
                    payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                else:
                    payload['state_dicts'][key] = value.state_dict()
        elif key in include_keys:
            payload['pickles'][key] = dill.dumps(value)

    - 如果属性中具有状态字典`state_dict`和加载状态字典方法`load_state_dict`(通常是模型或优化器等)，且不在`exclude_keys`中，就将其保存(cpu)
    - 如果属性在`include_keys`中，要使用`dill`序列化并保存到`payload['pickles']`

  - 保存检查点：

        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)

    - 依据`use_thread`的状态决定是否启用新线程保存检查点

  - 返回保存绝对路径：

        return str(path.absolute())

  - **`save_checkpoint`方法的作用:保存当前工作空间的检查点**

- `get_checkpoint_path`方法

        def get_checkpoint_path(self, tag='latest'):
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')

  - 参数:
    - `tag`: 标签，用于指定保存的检查点，默认为``'latest'``  

  - **作用:返回一个使用`pathlib`模块构建的指向检查点的路径**

- `load_payload`方法

        def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
            if exclude_keys is None:
                exclude_keys = tuple()
            if include_keys is None:
                include_keys = payload['pickles'].keys()

            for key, value in payload['state_dicts'].items():
                if key not in exclude_keys:
                    self.__dict__[key].load_state_dict(value, **kwargs)
            for key in include_keys:
                if key in payload['pickles']:
                    self.__dict__[key] = dill.loads(payload['pickles'][key])

  - 参数:
    - `payload`: 包含保存数据的字典
    - `exclude_keys`: 加载时要排除的属性(可选)
    - `include_keys`: 加载时额外包含的属性(可选)
    - `**kwargs`: 其他关键字参数，传递给`load_state_dict`方法

  - 处理`exclude_keys`和`include_keys`

        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

    - 如果未指定，`exclude_keys`默认为空，`include_keys`默认为数据字典中的序列化元素的键
  - 加载状态字典

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
            self.__dict__[key].load_state_dict(value, **kwargs)

    - 如果`payload['state_dicts']`中的键不被排除，就调用相应对象的`load_state_dict`方法加载状态字典
      - *`load_state_dict`方法是深度学习框架（如PyTorch）中模型或模块的一个方法，用于将序列化的状态字典加载到模型或模块中*
  - 加载序列化对象

        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])

    - 使用`dill`反序列化并赋值给对应的实例属性
  - **作用:将一个序列化的`payload`字典加载回当前实例的属性中**

- `load_checkpoint`方法

        def load_checkpoint(self, path=None, tag='latest',
                exclude_keys=None, 
                include_keys=None, 
                **kwargs):
            if path is None:
                path = self.get_checkpoint_path(tag=tag)
            else:
                path = pathlib.Path(path)
            payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
            self.load_payload(payload, 
                exclude_keys=exclude_keys, 
                include_keys=include_keys)
            return payload

  - 参数:
    - `path`: 检查点路径
    - `tag` : 检查点标签，默认为`'latest'`
    - `exclude_keys`: 加载时要排除的属性(可选)
    - `include_keys`: 加载时额外包含的属性(可选)
    - `**kwargs`: 其他关键字参数，传递给`torch.load`和`load_payload`方法
  - 路径处理

        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)

  - 加载检查点数据

        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)

    - 使用`torch.load`加载检查点文件，`dill`作为序列化模块，`'rb'`参数表示以二进制读取

  - 加载`payload`数据:

        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)

    - 调用前面的`load_payload`方法，将加载的数据应用到当前实例

  - 返回`payload`

        return payload

  - **作用:加载检查点数据**

- `create_from_checkpoint`类方法

        @classmethod
        def create_from_checkpoint(cls, path, 
                exclude_keys=None, 
                include_keys=None,
                **kwargs):
            payload = torch.load(open(path, 'rb'), pickle_module=dill)
            instance = cls(payload['cfg'])
            instance.load_payload(
                payload=payload, 
                exclude_keys=exclude_keys,
                include_keys=include_keys,
                **kwargs)
            return instance

  - `@classmethod`
    - 使用`@classmethod`装饰器定义一个类方法，可以通过类本身调用而不需要实例化，详情参阅[菜鸟教程:@classmethod修饰符](https://www.runoob.com/python/python-func-classmethod.html)

  - 参数:
    - `cls`: 类本身
    - `path`:检查点路径
    - `tag` : 检查点标签，默认为`'latest'`
    - `exclude_keys`: 加载时要排除的属性(可选)
    - `include_keys`: 加载时额外包含的属性(可选)
    - `**kwargs`: 其他关键字参数，传递给`load_payload`方法

  - 加载检查点数据

        payload = torch.load(open(path, 'rb'), pickle_module=dill)

  - 创建实例并加载数据

        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)

    - 使用加载的配置`payload`创建一个新实例，并将加载的数据应用到新实例中
  - 返回实例
  
        return instance

  - **作用： 从检查点创建一个新的实例**

- `save_snapshot`方法

        def save_snapshot(self, tag='latest'):

            path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
            path.parent.mkdir(parents=False, exist_ok=True)
            torch.save(self, path.open('wb'), pickle_module=dill)
            return str(path.absolute())
        
  - 参数
    - `tag` : 快照标签，默认为`'latest'`
  - **作用：将整个`BaseWorkSpace`实例序列化并保存到指定路径**

- `create_from_snapshot`类方法

        @classmethod
        def create_from_snapshot(cls, path):
            return torch.load(open(path, 'rb'), pickle_module=dill)

  - 参数
    - `cls`：类本身
    - `path`：指定快照文件路径
  
  - **作用：加载指定路径的快照文件**

### 辅助函数 `_copy_to_cpu`

        def _copy_to_cpu(x):
            if isinstance(x, torch.Tensor):
                return x.detach().to('cpu')
            elif isinstance(x, dict):
                result = dict()
                for k, v in x.items():
                    result[k] = _copy_to_cpu(v)
                return result
            elif isinstance(x, list):
                return [_copy_to_cpu(k) for k in x]
            else:
                return copy.deepcopy(x)

- 参数
  - `x`：输入数据，类型可为张量、字典和列表等
- 处理`torch.Tensor`

        if isinstance(x, torch.Tensor):
            return x.detach().to('cpu')

  - 如果输入数据是张量，就创建一个与当前张量共享内存但不需要计算梯度的新张量，将其移动到`cpu`上并返回

- 处理`dict`

        elif isinstance(x, dict):
            result = dict()
            for k, v in x.items():
                result[k] = _copy_to_cpu(v)
            return result

  - 如果输入数据是字典，就将每个值递归地传递给`_copy_to_cpu`处理，完成后返回新字典

- 处理`list`

        elif isinstance(x, list):
            return [_copy_to_cpu(k) for k in x]

  - 如果输入数据是列表，使用列表表达式递归处理并返回

- 处理其他类型

        else:
            return copy.deepcopy(x)

  - 如果数据是其他类型，就创建一个独立的副本并返回

### 功能

1. 使用`Hydra`和`OmegaConf`管理和访问配置参数
2. 保存和加载检查点

    - 支持保存当前状态的检查点，包括配置、模型状态字典和其他序列化对象。
    - 支持通过标签管理不同的检查点版本。
    - 提供线程化保存选项，避免阻塞主线程。

3. 保存和加载快照，支持保存和加载工作空间的完整状态，用于快速恢复，但假设代码不变，适用于研究中的快速实验。
4. `_copy_to_cpu`辅助函数，将复杂的数据结构中的张量从 GPU 移动到 CPU，以便序列化和保存
5. 允许通过`exclude_keys`和`include_keys`自定义保存和加载的内容，适应不同的需求和场景。

---

## Train diffusion unet hybrid workspace

    OmegaConf.register_new_resolver("eval", eval, replace=True)

- 通过`OmegaConf.register_new_resolver`方法，注册一个新的解析器，使得配置文件中能够动态解析`eval()`函数

### TrainDiffusionUnetHybridWorkspace(BaseWorkSpace)

**继承自BaseWorkSpace的派生类，用于管理训练（混合输入的）`Unet`网络**

**成员变量：**

        include_keys = ['global_step', 'epoch']

**方法：**

- `__init__`方法：
  - 构造函数初始化类，设置随即种子确保结果可复制

        def __init__(self, cfg: OmegaConf, output_dir=None):
            super().__init__(cfg, output_dir=output_dir)

            # set seed
            seed = cfg.training.seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    - `torch.manual_seed`,`np.random.seed`,`random.seed`分别为`torch`,`NumPy`和`Python`内置随机数生成器设置种子

  - 模型和`EMA`初始化

        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

    - 调用`hydra`库的`instantiate`函数根据`cfg.policy`实例化`DiffusionUnetHybridImagePolicy`
    - `EMA`（指数移动平均），一种加权移动平均方法，提高模型健壮性，默认为`None`
    - 若配置文件指定使用`EMA`，则将模型深拷贝赋值给`self.ema_model`，确保`self.model`和`self.ema_model`彼此独立

  - 优化器初始化

        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

    - 调用`hydra`库的`instantiate`函数根据`cfg.policy`实例化`optimizer`，并将`model`的参数传递给优化器

  - 初始化训练状态
  
        # configure training state
        self.global_step = 0
        self.epoch = 0

- `run`方法：
  - 启动训练循环。如果配置启用恢复选项，则从最近的检查点恢复训练

        def run(self):
            cfg = copy.deepcopy(self.cfg)

            # resume training
            if cfg.training.resume:
                lastest_ckpt_path = self.get_checkpoint_path()
                if lastest_ckpt_path.is_file():
                    print(f"Resuming from checkpoint {lastest_ckpt_path}")
                    self.load_checkpoint(path=lastest_ckpt_path)

    - *BaseWorkSpace中几个方法作用在这里体现*

  - 配置数据集

        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

    - 使用`hydra`配置数据集
    - 使用`Dataloader`加载训练数据
      - *`**cfg.dataloader`将`cfg.dataloader`中的键值对解包为关键字参数*
    - 获取数据归一化器

  - 配置验证数据集

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

    - 配置验证数据集
    - 使用`Dataloader`加载验证数据集

  - 对`model`和`ema_model`归一化处理

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

    - 确保输入数据的处理一致性

  - 配置学习率调度器

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )
    - 根据配置文件的参数控制学习率的变化

  - `ema_model`配置

        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

  - 运行环境配置

        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

    - 负责任务执行的模拟环境

  - 配置实验记录

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

    - 使用`wandb`配置实验记录，可视化训练过程中各种状态和参数

  - 检查点管理配置

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

    - 配置检查点管理器，用于保存训练过程中最好的检查点

  - 将设备和优化器迁移至GPU

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

  - `Debug`配置

        if cfg.training.debug:
        cfg.training.num_epochs = 2
        cfg.training.max_train_steps = 3
        cfg.training.max_val_steps = 3
        cfg.training.rollout_every = 1
        cfg.training.checkpoint_every = 1
        cfg.training.val_every = 1
        cfg.training.sample_every = 1

  - 训练过程

        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

    - 对于每个循环批次
      - 将每个批次数据移动到`GPU`
      - 计算损失，反向传播梯度
      - 满足梯度累积步数时，优化器更新权重，重置梯度
      - 若启用了`EMA`，更新`EMA`模型

  - 记录日志及损失累积

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

    - 记录损失和学习率，并将其记录到`wandb`和`json_logger`中
      - `tepoch.set_postfix`设置进度条的后缀
      - 如果达到设置的循环步数，退出循环

  - 每轮训练结束：验证、`rollout`
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

    - 计算平均训练损失
    - 使用`policy`变量指向`model`，将其设置为评估模式
    - 运行`rollout`并记录
    - 运行验证，将验证的平均损失记录
