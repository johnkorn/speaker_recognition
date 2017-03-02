
import React from 'react';
import View from '../components/Train';
import { connect } from 'react-redux';
import { train } from '../api/verifier';
import {send, complete as trains} from '../actions/train';
import { complete as users } from '../actions/users';

export default connect((state) => ({
  isSend: state.train.send
}), (dispatch) => ({
  onTrain(userId, files) {
    dispatch(send());
    train(userId, files)
      .then(
        data => {
          dispatch(users(null, data))
        }
      )
      .then(
        data => dispatch(trains()),
        err => dispatch(trains(err))
      )
  }
}))((props) => (
  <View
    isSend={props.isSend}
    onClick={(userId, files) => props.onTrain(userId, files)}
  />
));