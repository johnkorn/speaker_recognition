
import React from 'react';
import View from '../components/ResetClassifier';
import { connect } from 'react-redux';
import { reset } from '../api/verifier';
import { send, complete } from '../actions/reset';
import { clear } from '../actions/users';

export default connect((state) => ({
  isSend: state.reset.send
}), (dispatch) => ({
  onReset() {
    dispatch(send());
    reset()
      .then(
        data => dispatch(complete()),
        err => dispatch(complete(err))
      )
      .then(
        () => dispatch(clear())
      );
  }
}))((props) =>
  <View
    isSend={props.isSend}
    onClick={() => props.onReset()}
  />
);