from sklearn.ensemble import GradientBoostingRegressor


class Model(GradientBoostingRegressor):
    """
    Дополнительный уровень абстракции над используемой моделью.
    Может включать в себя некоторую дополнительную логику, необходимую в скриптах обучения или
    инференса.
    """

    pass
