
import React from 'react';
import {connect} from 'react-redux';
import View from '../components/Layout';

export default connect((state) => ({
  users: state.users,
}))((props) => (
  <View
    isSend={props.users.isSend}
    users={props.users.data}
  >
    {props.children}
  </View>
));