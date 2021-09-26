from ethereum.tools import tester

c = tester.Chain()
x = c.contract("""
def foo(x):
    return x + 5
""", language='serpent')
assert x.foo(2) == 7
bn = c.head_state.block_number
c.mine(5)
assert c.head_state.block_number == bn + 5
x2 = c.contract("""
data moose

def increment_moose():
    self.moose += 1
    return self.moose
""", language='serpent')
assert x2.increment_moose() == 1
assert x2.increment_moose() == 2