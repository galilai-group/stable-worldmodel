class WorldModel(nn.Module):

    # sanity check (shape, types)
    # log prediction time etc..

    def __init__(self, wm):
        super().__init__()
        self.wm = wm


    def encode(self):
        embedding = ...
        return embedding

    def predict(self, obs, actions, timestep):
        """predict next s_t+H embedding given s_t + action sequence
        i.e rollout the dynamics model for H steps
        """
        predicted_embedding = ...
        return predicted_embedding

def decode_rollout():
    # if decoder
    pass