
import React, { Component, PropTypes } from 'react';
import Timer from './Timer';
import '../resources/button.css';

export default class Train extends Component {

  static propTypes = {
    onClick: PropTypes.func.isRequired,
    isSend: PropTypes.bool.isRequired
  };

  state = {
    userId: null,
    files: []
  };

  _onClick = () => {
    this.props.onClick(
      this.state.userId,
      this.state.files
    );
  };

  _onChangeUserId = (value) => {
    this.setState({userId: value});
  };

  render() {
    return (
     	<div>
        <p>
          Upload multiple files for trainig (new or existing user):<br/>
          Input user id for training (from the list above).<br/>
          Or just input a name for new user!
        </p>
        <input
          type="number"
          value={this.state.user_id}
          name="userid"
          min="1"
          step="1"
          onChange={(e) => {
            this._onChangeUserId(parseInt(e.target.value, 0));
          }}
        />
        <Timer
          max={10}
          long={5}
          onAdd={(filename, blob) => {
            this.setState({
              files: this.state.files.concat([{
                key: filename,
                data: blob
              }])
            });
          }}
          onDelete={(id) => {
            this.setState({
              files: this.state.files.filter(_ => _.key !== id),
            });
          }}
        />
        <p> Send user: </p>
        <p style={
          {
            display: (this.state.files.length > 1 && parseInt(this.state.userId, 0) > 0) ? 'initial' : 'none'
          }
        }>
          <input
            type="button"
            value="Upload files"
            className="button"
            disabled={this.props.isSend}
            onClick={() => this._onClick()}
          />
        </p>
	    </div>
    );
  }

}