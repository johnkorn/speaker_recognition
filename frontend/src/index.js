import React from 'react';
import ReactDOM from 'react-dom';
import App from './layout/components/App';
import PageNotFound from './layout/components/PageNotFound';
import './index.css';
import { Router, Route, browserHistory, IndexRedirect } from 'react-router';
import reducers from './verifier/reducers/index';
import LayoutView from './verifier/containers/LayoutView';
import ResetView from './verifier/containers/ResetView';
import TrainView from './verifier/containers/TrainView';
import VerifyView from './verifier/containers/VerifyView';
import { getUsers } from './verifier/api/verifier';
import { send, complete } from './verifier/actions/users';
import { createStore, combineReducers } from 'redux'
import { Provider } from 'react-redux'
import { syncHistoryWithStore, routerReducer } from 'react-router-redux'

const store = createStore(
  combineReducers({
    ...reducers,
    routing: routerReducer
  })
);

const history = syncHistoryWithStore(browserHistory, store);

store.dispatch(send());
getUsers()
  .then(data => store.dispatch(complete(null, data)))
  .then(() => ReactDOM.render(
    <Provider store={store}>
      <Router history={history}>
        <Route path="/" component={App}>
          <IndexRedirect to="/train" />
          <Route path="/" component={LayoutView}>
            <Route path="reset" component={ResetView}/>
            <Route path="train" component={TrainView}/>
            <Route path="verify" component={VerifyView}/>
            <Route path='*' component={PageNotFound} />
          </Route>
        </Route>
      </Router>
    </Provider>,
    document.getElementById('root')
  ))
  .catch(err => store.dispatch(complete(err)));
