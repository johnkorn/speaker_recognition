
import React from 'react';
import View from '../components/Verify';
import { connect } from 'react-redux';
import { verify } from '../api/verifier';
import { send, complete } from '../actions/verify';

export default connect((state) => ({
  isSend: state.verify.send,
  speaker: state.verify.speaker
}), (dispatch) => ({
  onVerify(file) {
    dispatch(send());
    verify(file)
      .then(
        data => dispatch(complete(null, data.verify)),
        (err) => dispatch(complete(err))
      )
  }
}))((props) => (
  <View
    isSend={props.isSend}
    speaker={props.speaker}
    onClick={(file) => props.onVerify(file)}
  />
));