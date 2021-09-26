from unit_commitment.test_cases.case24 import case24
from pypower.runopf import runopf
case=case24()
runopf(case)