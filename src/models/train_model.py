

def build_random_forrest(n_estimators=100):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=n_estimators)
