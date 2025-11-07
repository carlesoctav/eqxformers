TrainingLoop
    run_name:
    tracker: tracker
    metrics: Metrics
    callback: [Callback]

    abstractmethod
    def loss_function(self, model, batch, *, **batch_kwargs, key) -> (loss, aux)


    @abstractmethod
    def train_step(self, state, batch, *, **batch_kwargs, key) -> 



    def make_train_state() -> train_state



    def load_from_checkpoint() -> train_state



    train(state, train_loader):
        do the train


    benchmark(state, train_loader):
        do benchmark and aggresive logging and jax.wait until ready



state = TrainState


callback take stepInfo ->
    state
    loss
    model
    opt_state
    batch

TrainState:
    step
    model
    optimizer
    opt_state
    key


app.trainer(xx, yy)
