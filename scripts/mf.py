"""Explicit observed-rating biased MF, fitted by ridge alternating least squares.

Objective: sum_observed (r-mu-bu-bi-pu@qi)^2 + reg*(||b||²+||P||²+||Q||²).
Unobserved pairs are absent from the objective. Global mean is fixed from train.
"""
import copy
import numpy as np
import pandas as pd


class MatrixFactorization:
    def __init__(self, factors=8, reg=1., epochs=20, seed=42):
        if not isinstance(factors, int) or factors < 0 or not isinstance(epochs, int) or epochs < 1:
            raise ValueError('Use nonnegative integer factors and positive epochs')
        if not np.isfinite(reg) or reg <= 0:
            raise ValueError('reg must be finite and positive')
        self.factors, self.reg, self.epochs, self.seed = factors, float(reg), epochs, seed

    @staticmethod
    def _validate(frame):
        if frame.empty or frame[['user_id','place_id']].isna().any().any():
            raise ValueError('Need nonempty observed pairs with IDs')
        ratings = frame.rating.to_numpy(float)
        if not np.isfinite(ratings).all() or not ((ratings>=1)&(ratings<=5)).all():
            raise ValueError('Observed ratings must be finite in [1,5]')
        if frame.duplicated(['user_id','place_id']).any():
            raise ValueError('Aggregate repeated ratings within the split first')

    def _solve(self, factors, residual):
        x = np.column_stack([np.ones(len(residual)), factors])
        return np.linalg.solve(x.T@x + self.reg*np.eye(self.factors+1), x.T@residual)

    def fit(self, data):
        train = data['train'].sort_values(['user_id','place_id']).reset_index(drop=True)
        self._validate(train)
        self.users = np.array(sorted(train.user_id.unique()))
        self.items = np.array(sorted(train.place_id.unique()))
        self.user_idx = pd.Series(np.arange(len(self.users)), index=self.users)
        self.item_idx = pd.Series(np.arange(len(self.items)), index=self.items)
        u = self.user_idx.loc[train.user_id].to_numpy()
        i = self.item_idx.loc[train.place_id].to_numpy()
        r = train.rating.to_numpy(float)
        self.mu = float(r.mean())
        rng = np.random.default_rng(self.seed)
        self.P = rng.normal(0,.1,(len(self.users),self.factors))
        self.Q = rng.normal(0,.1,(len(self.items),self.factors))
        self.bu, self.bi = np.zeros(len(self.users)), np.zeros(len(self.items))
        # Group only observed coordinates, without creating a dense rating matrix.
        user_groups = [np.flatnonzero(u==j) for j in range(len(self.users))]
        item_groups = [np.flatnonzero(i==j) for j in range(len(self.items))]
        self.loss_history = []
        for _ in range(self.epochs):
            for j, pos in enumerate(user_groups):
                z = self._solve(self.Q[i[pos]], r[pos]-self.mu-self.bi[i[pos]])
                self.bu[j], self.P[j] = z[0], z[1:]
            for j, pos in enumerate(item_groups):
                z = self._solve(self.P[u[pos]], r[pos]-self.mu-self.bu[u[pos]])
                self.bi[j], self.Q[j] = z[0], z[1:]
            errors = r-self.mu-self.bu[u]-self.bi[i]-np.einsum('ij,ij->i',self.P[u],self.Q[i])
            penalty = sum(np.square(x).sum() for x in (self.P,self.Q,self.bu,self.bi))
            self.loss_history.append(float(errors@errors+self.reg*penalty))
        return self

    def score(self, users, candidates):
        ur = self.user_idx.reindex(np.asarray(users)); ir = self.item_idx.reindex(np.asarray(candidates))
        up = np.flatnonzero(ur.notna()); ip = np.flatnonzero(ir.notna())
        result = np.full((len(users),len(candidates)), -np.inf)
        if len(up) and len(ip):
            ui, ii = ur.iloc[up].to_numpy(int), ir.iloc[ip].to_numpy(int)
            result[np.ix_(up,ip)] = self.mu+self.bu[ui,None]+self.bi[ii][None,:]+self.P[ui]@self.Q[ii].T
        return result

    def predict(self, users, places):
        if len(users)!=len(places):
            raise ValueError('Prediction pairs must align')
        ur = self.user_idx.reindex(np.asarray(users)); ir = self.item_idx.reindex(np.asarray(places))
        mask = ur.notna().to_numpy() & ir.notna().to_numpy()
        out = np.full(len(users),np.nan)
        ui, ii = ur[mask].to_numpy(int), ir[mask].to_numpy(int)
        out[mask] = self.mu+self.bu[ui]+self.bi[ii]+np.einsum('ij,ij->i',self.P[ui],self.Q[ii])
        return out

    def with_history(self, support):
        """New-user fold-in: fixed global mean/item factors/biases; support only.

        Unknown support items are ignored. With no known item, user is unsupported.
        A new object is returned; the fitted model is never changed.
        """
        self._validate(support)
        model = copy.copy(self)
        known = support[support.place_id.isin(set(self.items))]
        users, vectors = [], []
        for user, group in known.groupby('user_id',sort=True):
            ii = self.item_idx.loc[group.place_id].to_numpy(int)
            z = self._solve(self.Q[ii],group.rating.to_numpy(float)-self.mu-self.bi[ii])
            users.append(user); vectors.append(z)
        model.users = np.asarray(users)
        model.user_idx = pd.Series(np.arange(len(users)),index=users,dtype=int)
        z = np.asarray(vectors).reshape(len(users),self.factors+1)
        model.bu, model.P = z[:,0], z[:,1:]
        return model
