class Config(object):
    """
    Configuration stub.

    Configurations should return an object without any additional
    input. 

    I'm calling a "parameterized" configuration file a "generator",
    So the configuration may call a generator after providing additional
    information.
    """

    def build(self):
        raise NotImplementedError()
