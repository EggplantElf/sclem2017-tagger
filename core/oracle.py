from data import *

class Oracle(object):
    def __init__(self, system):
        self.system = system

    def tell(self, state):
        for action in self.system.actions:
            if self.can_do(state, action):
                return self.system.get_action_idx(action)
        return -1

    def can_parse(self, sent):
        state = State(sent)
        while not state.finished():
            y = self.tell(state)
            if y == -1:
                return False
            else:
                state = state.perform(self.system, y)
        return True

    def can_do(self, state, action):
        # it has to be at least valid
        if not state.valid(action):
            return False
        # constraints by the oracle
        if action == 'LA':
            return state.sent.gold_heads[state.stk[-2]] == state.stk[-1] \
                    and self.has_all_deps(state, state.stk[-2])
        elif action == 'RA':
            return state.sent.gold_heads[state.stk[-1]] == state.stk[-2] \
                    and self.has_all_deps(state, state.stk[-1])
        elif action == 'LA2':
            return state.sent.gold_heads[state.stk[-3]] == state.stk[-1] \
                    and self.has_all_deps(state, state.stk[-3])
        elif action == 'RA2':
            return state.sent.gold_heads[state.stk[-1]] == state.stk[-3] \
                    and self.has_all_deps(state, state.stk[-1])
        elif action == 'LA3':
            return state.sent.gold_heads[state.stk[-4]] == state.stk[-1] \
                    and self.has_all_deps(state, state.stk[-4])
        elif action == 'RA3':
            return state.sent.gold_heads[state.stk[-1]] == state.stk[-4] \
                    and self.has_all_deps(state, state.stk[-1])
        elif action == 'SW':
            return (not state.sent.is_projective) and self.check_order(state) and self.check_mpc(state)
        elif action == 'SH':
            return True
        else:
            return False

    # not quite accurate, should check the items instead of the sum
    def has_all_deps(self, state, head):
        return state.sent.gold_heads.count(head) == sum(1 for d,h,l in state.arcs if h == head)

    def check_order(self, state):
        order = state.sent.inorder
        return order[state.stk[-2]] > order[state.stk[-1]]


    def check_mpc(self, state):
        if not state.sent.mpc:
            self.get_mpc(state.sent)
        return not state.bfr or state.sent.mpc[state.stk[-1]] is not state.sent.mpc[state.bfr[0]]

    def get_mpc(self, sent):
        # assert state is initial state
        mpc = sent.mpc = [[i] for i in range(len(sent.tokens))]
        state = State(sent)
        while True:
            if self.can_do(state, 'LA'):
                merged = mpc[state.stk[-1]] + mpc[state.stk[-2]]
                for i in merged:
                    mpc[i] = merged
                state = state.perform(self.system, self.system.get_action_idx('LA'))
            elif self.can_do(state, 'RA'):
                merged = mpc[state.stk[-1]] + mpc[state.stk[-2]]
                for i in merged:
                    mpc[i] = merged
                state = state.perform(self.system, self.system.get_action_idx('RA'))
            elif self.can_do(state, 'SH'):
                state = state.perform(self.system, self.system.get_action_idx('SH'))
            else:
                return


