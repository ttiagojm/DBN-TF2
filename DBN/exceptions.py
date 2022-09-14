class MismatchShape(Exception):
   def __init__(
        self,
        shape_1,
        shape_2,
        message=None,
    ):
        self.message = (
            f"Shapes are different and they should be equal. We have shape {shape_1} and shape {shape_2}"
            if message is None
            else message
        )
        super().__init__(self.message) 

class MismatchCardinality(Exception):
    def __init__(
        self,
        type,
        message=None,
    ):
        self.message = (
            f"{type} doesn't has the required cardinality. Check annotation."
            if message is None
            else message
        )
        super().__init__(self.message)


class NonSquareInput(Exception):
    def __init__(
        self,
        message="The input is not square (row size different from col size)",
    ):
        self.message = message
        super().__init__(self.message)
