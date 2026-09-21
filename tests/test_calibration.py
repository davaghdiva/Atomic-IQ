from pathlib import Path
import json


def test_selected_calibration():
    root=Path(__file__).resolve().parents[1]
    p=root/'RESULTS'/'calibration'/'selected_parameters.json'
    assert p.exists()
    z=json.loads(p.read_text())
    a=z['Atomic-IQ']['parameters']
    assert a=={'c':0.4,'delta':1.25,'p':0.1,'chi':0.9}
    assert abs(z['GS_STAR']['threshold']-.92)<1e-12
    assert abs(z['SRE_STAR']['alpha']-0.0)<1e-12
    assert not any('chi1' in k.lower() for k in z)
