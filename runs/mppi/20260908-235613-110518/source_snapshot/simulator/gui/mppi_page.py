"""The application's sole offline planner; shared saved-result infrastructure."""
from .cem_page import CEMPage


class MPPIPage(CEMPage):
    def __init__(self, root):
        super().__init__(root, optimizer='mppi')
