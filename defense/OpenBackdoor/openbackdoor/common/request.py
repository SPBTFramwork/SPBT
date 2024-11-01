class Request:
    def __init__(self, config_json):
        for key, value in config_json.items():
            setattr(self, key, value)
        self.task2lang = {'Defect': 'c', 'Clone': 'java', 'Translate': 'java_cpp'}
    
    def getTask(self):
        return self.task
    
    def getInputKey(self):
        return self.input_key
    
    def getOutputKey(self):
        return self.output_key

    def getDefenseType(self):
        return self.defense_type

    def getCleanDatasetPath(self):
        /home/nfs/share/backdoor2023/defense/OpenBackdoor/datasets/defect/c
        return self.clean_dataset_path