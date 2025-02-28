import React from 'react';
import ReactDOM from 'react-dom';
import { HashRouter, Route, Switch, Redirect } from 'react-router-dom';
import { Provider } from 'react-redux';
import store from 'store/store';

import Admin from 'layouts/Admin.js';
import ProjectLayout from 'layouts/ProjectLayout.js';
import FlowLayout from 'layouts/FlowLayout';

ReactDOM.render(
  <Provider store={store}>
    <HashRouter>
      <Switch>
        <Route
          path="/projects/:projectId/flows/:flowId"
          component={FlowLayout}
        />
        <Route path="/admin" component={Admin} />
        <Route path="/projects/:projectId" component={ProjectLayout} />
        <Redirect from="/" to="/admin/main" />
      </Switch>
    </HashRouter>
  </Provider>,
  document.getElementById('root')
);
