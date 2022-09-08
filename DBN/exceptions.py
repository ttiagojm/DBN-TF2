class RBMListEmpty(Exception):
    def __init__(
        self,
        message="RBMs list is empty. You must pass a list of RBMBeroulli/RBMConv",
    ):
        self.message = message
        super().__init__(message)
