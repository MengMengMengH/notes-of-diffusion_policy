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
    - [ ] 待做：解读`self.model.compute_loss`（计算损失）方法（在[`diffusion_unet_hybrid_image_policy`](diffusion_policy/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py)文件中定义）

  - 采样与动作误差评估

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']

                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

    - 对训练集批次进行扩散采样，并且计算预测动作和真实动作的均方误差（`MSE`）
    - [ ] 待做：解读`policy.predict_action`（预测动作）方法（在[`diffusion_unet_hybrid_image_policy`](diffusion_policy/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py)文件中定义）

  - 保存检查点

                if (self.epoch % cfg.training.checkpoint_every) == 0:

                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

    - 调用父类的保存检查点和快照方法
      - *注意`save_point`使用了多线程，可能会导致保存检查点的操作还未完成时，代码已经继续往下执行，由于保存检查点的操作可能仍在进行中，因此检查点文件（模型状态、训练状态等）的内容可能尚未完全写入到磁盘，如果此时试图立即复制或使用最新的检查点文件，文件可能会处于不完整或空白状态*

  - 结束当前周期并更新日志、状态

        policy.train()
        wandb_run.log(step_log, step=self.global_step)
        json_logger.log(step_log)
        self.global_step += 1
        self.epoch += 1

### `@hydra.main`装饰器

    @hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)

- `@hydra.main`是`Hydra`提供的一个装饰器，用于将函数`main`转换为`Hydra`应用程序。
- `version_base=None`表示不使用特定的版本控制。
- `config_path`指定配置文件的位置，使用`pathlib`动态构造路径，指向当前脚本的父级目录中的 config 文件夹。
- `config_name`设置为当前脚本的名称（去掉扩展名），即根据脚本的命名来确定要加载的配置文件。

### `main`函数定义

    def main(cfg):
        workspace = TrainDiffusionUnetHybridWorkspace(cfg)
        workspace.run()

- 参数：
  - `cfg`:由`Hydra`加载的配置对象
- 在函数内部，创建一个`TrainDiffusionUnetHybridWorkspace`的实例，传入加载的配置 `cfg`

***这一部分中最重要的是初始化过程中的`self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)`，此行决定了模型的性质。因此下一步将研究`DiffusionUnetHybridImagePolicy`类，由于继承关系是`DiffusionUnetHybridImagePolicy`$\rightarrow$`BaseImagePolicy`$\rightarrow$`ModuleAttrMixin`$\rightarrow$`torch.nn.Module`，下文将先从`ModuleAttrixin`开始***

## module attr mixin

### `ModuleAttrMixin`类

**继承自`torch.nn`的派生类，为了方便获取模型中参数的设备类型 `device` 和数据类型 `dtype`，使得在模型的开发和调试中能够更快速地访问这些信息**

- 初始化方法

            def __init__(self):
                super().__init__()
                self._dummy_variable = nn.Parameter()

  - 创建了一个空的`nn.Parameter`变量`_dummy_variable`，确保该模块至少有一个参数，以便在后续的`device`和`dtype`属性的实现中使用`parameters`函数
    - 为什么需要`self._dummy_variable`：在某些模型或自定义模块中，可能没有实际的`nn.Parameter`。如果没有参数，`self.parameters()`将返回一个空的迭代器。在这种情况下，`device`和`dtype`属性的`next(iter(self.parameters()))`会导致`StopIteration`异常。因此，通过定义一个虚拟参数来避免这个问题。

- 获取设备属性

            @property
            def device(self):
                return next(iter(self.parameters())).device

  - `@property`装饰器将`device`定义为一个只读属性，允许直接通过`model.device`访问
  - `next(iter(self.parameters())).device`的含义是从模型的参数迭代器`self.parameters()`中获取第一个参数，并访问其`device`属性

- 获取数据类型属性

            @property
            def dtype(self):
                return next(iter(self.parameters())).dtype

  - `@property`装饰器将`dtype`定义为一个只读属性，允许直接通过`model.dtype`访问
  - `next(iter(self.parameters())).dtype`的含义是从模型的参数迭代器`self.parameters()`中获取第一个参数，并访问其`dtype`属性

## base image policy

### BaseImagePolicy

**继承自`ModuleAttrMixin`的抽象基类，通常用于定义一系列方法和接口供具体的策略模型实现（例如卷积网络、变分自编码器等图像策略模型）**

- `predict_action`方法

        def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """
            obs_dict:
                str: B,To,*
            return: B,Ta,Da
            """
            raise NotImplementedError()

  - 该方法设计目的是为模型的策略推理接口提供一个标准定义
  - 参数解释：
    - `obs_dict`：输入字典`Dict[str,torch.Tensor]`，其中每个键`str`表示一种观测（图像、状态），每个值`torch.tensor`是一个张量，表示输入数据
      - 形状约定：注释中指出每个张量的形状格式为`(B,To,*)`
        - `B`：批次大小
        - `To`：观测时间维度
        - `*`：任意形状的其余维度
  - 返回值：
    - 返回一个字典，表示预测的动作`action`
      - 形状约定：
        - `B`：批次大小
        - `Ta`：动作执行时间维度
        - `Da`：动作维度
  - `NotImplementedError`**异常**：该方法目前没有实际实现，子类（例如实际的图像策略模型）必须实现该接口，定义其具体的动作预测逻辑

- `reset`方法

        def reset(self):
            pass

  - `reset()`方法用于重置策略模型的内部状态
  - 在某些强化学习或序列模型中，策略模型可能是有状态的`（stateful）`，这意味着它们在不同时间步之间保留内部状态`（例如 RNN、LSTM）`。在这种情况下，每次开始新的一轮推理时（例如新一轮游戏或新一组序列数据）需要调用`reset()`以清除之前的状态
  -当前实现是一个空函数`（pass）`，表示默认情况下策略模型是无状态的`（stateless）`。子类可以根据具体需求重写该方法，重置模型内部状态（如隐藏层状态、注意力权重等）。

- `set_normalizer`方法

        def set_normalizer(self, normalizer: LinearNormalizer):
            raise NotImplementedError()

  - 该方法设计为一个接口，用于设置模型的输入/输出正则化器`Normalizer`
  - 参数：
    - `normalizer`：类型为`LinearNormalizer`，表示一个线性正则化对象
    - `LinearNormalizer`的作用通常是对输入的观测值进行标准化`normalization`或去标准化`denormalization`，例如将输入图像数据标准化到`[0, 1]`范围内，或者对动作值进行均值和方差的调整。

- `NotImplementedError`**异常**：该方法目前没有实际实现，子类必须实现该接口以定义如何应用正则化器到模型中。通常，模型需要正则化器来保持输入数据的标准一致性，从而提升模型训练的稳定性和推理的准确性。

- `BaseImagePolicy`的设计目的是为所有基于图像的策略模型提供一个统一的接口和基础结构。在实际应用中，不同类型的图像策略模型（如卷积神经网络、基于注意力的模型等）可能有不同的架构，但它们都应具备以下能力：
  - 动作预测`predict_action`：能够从给定的观测中预测相应的动作。
  - 状态重置`reset`：在需要时，能够重置其内部状态（如有）。
  - 正则化`set_normalizer`：能够使用标准化器对输入和输出数据进行处理。

## Diffusion Unet hybrid image policy

### DiffusionUnetHybridImagePolicy

**该类使用扩散模型（diffusion models）来处理条件`conditional`图像输入，并结合*行为克隆`Behavior Cloning`*和*图像特征编码器*，以预测机器人在多步操作序列中的动作。这个模型框架综合了`robomimic`框架的图像处理模块和`diffusers`扩散模型，设计用于解决多步序列的动作推理任务。**

- 类的初始化构造与组建
  - 构造函数参数：

        def __init__(self, 
                    shape_meta: dict,
                    noise_scheduler: DDPMScheduler,
                    horizon, 
                    n_action_steps, 
                    n_obs_steps,
                    num_inference_steps=None,
                    obs_as_global_cond=True,
                    crop_shape=(76, 76),
                    diffusion_step_embed_dim=256,
                    down_dims=(256,512,1024),
                    kernel_size=5,
                    n_groups=8,
                    cond_predict_scale=True,
                    obs_encoder_group_norm=False,
                    eval_fixed_crop=False,
                    **kwargs):

    - `shape_meta`：表示输入数据（观察值和动作）的形状元数据，用于指定动作和观察数据的维度结构
    - `noise_scheduler`：一个用于时间步长调度的扩散噪声调度器对象`DDPMScheduler`，用于模拟扩散过程
    - `horizon`：表示模型能够预测的最长动作序列的长度
    - `n_action_steps`：表示一次推理中需要预测的动作步数
    - `n_obs_steps`：表示每次输入推理中包含的观测步数
    - `num_inference_steps`：用于控制扩散模型中推理过程的时间步数
    - `obs_as_global_cond`：布尔类型参数，决定是否将观测特征作为全局条件输入，默认开启
    - `crop_shape`：元组，定义了处理图像时的裁减尺寸
    - `diffusion_step_embed_dim`：指定扩散步骤嵌入的维度
    - `down_dims`：定义了处理数据时降维尺寸
    - `kernels_size`：定义了卷及核的大小
    - `n_groups`：定义了分组卷积的组数
    - `cond_predict_scale`：指示是否对条件预测进行缩放
    - `obs_encoder_group_norm`：指示是否在观察编码器中使用组归一化
    - `eval_fixed_crop`：指示在评估时是否使用固定的裁减
    - `**kwargs`：一个字典，包含其他任意数量的关键字参数，用于提供额外的配置选项

- 初始化观测处理器

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        obs_shape_meta = shape_meta['obs']
        obs_config = {'low_dim': [], 'rgb': [], 'depth': [], 'scan': []}

        # process obs shape meta to update obs_config
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            type = attr.get('type', 'low_dim')
            obs_key_shapes[key] = list(shape)
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)

  - 解析`shape_meta`中提供的动作和观测数据的元数据，并且基于类型(`low_dim`,`rgb`等)将观测数据分类
  - 初始化`obs_config`，用于指定`robomimic`中的观测模式，以匹配输入数据的格式
    - *`dict.get()`是`python`中字典获取键对应的值，如果键不存在，则用第二个参数替代值返回*

- 配置`robomimic`中的观测数据模块

        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')

        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        ObsUtils.initialize_obs_utils_with_config(config)

  - `get_robomimic_config`：获取一个基于`robomimic`的行为克隆模型配置 (`bc_rnn`)，该配置会根据`obs_config`来调整模型的输入模式（如使用图像观测或低维观测）
  - **设置图像裁剪参数**：根据`crop_shape`是否为空，决定是否应用`CropRandomizer`模块。`CropRandomizer`模块用于在训练时对图像进行随机裁剪，以增强数据的多样性。这样可以帮助模型更好地处理不同视角下的图像输入
  - 根据`config`配置的观测模式（如`rgb`图像`low_dim`低维观测数据）来初始化`ObsUtils`的内部变量，使得观测数据能够被正确地预处理和解析

- 创建`robomimic`策略模型并提取观测编码器

        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )
        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']

  - 使用`algo_factory`根据`robomimic`配置创建策略模型，并从中提取观测数据编码器 (obs_encoder)
  - `obs_encoder`是一个神经网络模块，负责将原始观测数据（图像或低维状态）编码成高维特征，用于进一步的策略预测

- 替换`obs_encoder`中的`BatchNorm`层（可选）

        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )

  - 这段代码根据用户传入的参数`obs_encoder_group_norm`，决定是否将`BatchNorm`层替换为`GroupNorm`。这种替换在小批次训练时尤其有用，可以减少 BatchNorm 的噪声影响

- 替换`obs_encoder`模块中的所有`CropRandomizer`子模块（可选）

        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )
  - 当`eval_fixed_crop`为`True`时，遍历`obs_encoder`中的所有子模块，将所有类型为`rmbn.CropRandomizer`的实例替换为`dmvc.CropRandomizer`的新实例
  - 这个替换通常是为了在评估阶段使用不同的裁剪逻辑或参数，确保模型在评估时的稳定性和一致性

- 初始化扩散模型（`diffusion_model`）

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

  - `obs_feature_dim = obs_encoder.output_shape()[0]`，获取观察编码器`(obs_encoder)`输出的特征维度`（obs_feature_dim）`，通常这是在处理输入观测数据后生成的特征数量。
  - `input_dim = action_dim + obs_feature_dim`，`input_dim`定义了输入到扩散模型中的维度。在这里，输入维度是由动作维度`（action_dim）`和观察特征维度`（obs_feature_dim）`的和构成的。
  - `global_cond_dim = None`，初始化`global_cond_dim`为`None`，这是后续用于条件输入的维度
  - 判断是否将观察作为全局条件进行输入
    - 若是，则只将动作维度作为输入维度，`input_dim`设置为`action_dim`，`global_cond_dim`设为观察特征维度乘以观察步数`obs_feature_dim*n_obs_steps`，这表示全局条件的维度
  - `ConditionalUnet1D`：基于条件`UNet`结构的扩散模型，它接收观测特征作为条件输入，并通过时间步长嵌入（`diffusion_step_embed_dim`）和`UNet`网络层进行条件扩散和动作预测
    - 模型参数：
      - `input_dim`: 输入的特征维度
      - `local_cond_dim=None`: 局部条件的维度，这里设为`None`，可能表示没有使用局部条件
      - `global_cond_dim`: 全局条件的维度
      - `diffusion_step_embed_dim`: 扩散步嵌入的维度
      - `down_dims`: 网络中每个下采样层的维度列表
      - `kernel_size`: 卷积核的大小
      - `n_groups`: 分组归一化的组数
      - `cond_predict_scale`: 条件预测缩放参数

- 初始化其他组件

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )

  - `self.obs_encoder = obs_encoder`: 保存观察编码器
  - `self.model = model`: 保存创建的扩散模型
  - `self.noise_scheduler = noise_scheduler`: 保存噪声调度器
  - `self.mask_generator = LowdimMaskGenerator(...)`: 初始化一个低维掩码生成器，参数包括动作维度、观察维度、最大观察步数等。这个生成器用于生成掩码，控制模型在训练时应该关注哪些输入。
    - `LowdimMaskGenerator`类的主要功能是根据输入的参数生成一个掩码（mask），用于控制在序列模型中哪些特征（观察或动作）在某个时间步上被看见或隐藏。这个类的典型应用场景是扩散模型或其他序列生成模型的训练中，用来限制模型在不同时间步上能访问的输入特征，尤其是在涉及行为建模或时序数据处理时

- 正则化器和其他参数

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

  - `self.normalizer = LinearNormalizer()`: 初始化一个线性归一化器，用于对输入进行归一化处理
  - 其他参数如`horizon`、`obs_feature_dim`、`action_dim`等用于保存模型的内部状态

- 设置推理步骤

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

  - 这段代码用于设置推理时的步骤数，如果未提供`num_inference_steps`，则默认为噪声调度器的训练时间步数
