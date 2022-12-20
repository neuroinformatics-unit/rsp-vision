class DataRaw:
    def __init__(self, data: dict, is_allen: bool = True):
        if is_allen:
            self.day = data["day"]
            self.f = data["f"]
            self.imaging = data["imaging"]
            self.is_cell = data["is_cell"]
            self.r_neu = data["r_neu"]
            self.stim = data["stim"]
            self.trig = data["trig"]
        else:
            raise NotImplementedError("Only loading for Allen data is implemented")
