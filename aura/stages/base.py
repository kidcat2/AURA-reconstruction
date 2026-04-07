from abc import ABC, abstractmethod

class BaseStage(ABC):

    """
    config : 경로, 파라미터, 설정 
    context : 이전 Stage의 결과물
    """
    @abstractmethod
    def run(self, config, context):
        pass
