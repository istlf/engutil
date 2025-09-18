import engutil

def test_import():
    assert hasattr(engutil, "generate_sine")

def test_generate_sine():
    t, x = engutil.generate_sine(10, 0, 1, fs=100)
    assert len(t) == len(x)
    assert max(x) <= 1.0 and min(x) >= -1.0
