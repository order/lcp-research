import config

class ValueProcessor(config.Processor):
    def process(self,data):
        assert('value' in data)
        return data['value']
