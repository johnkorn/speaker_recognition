import React, { Component, PropTypes } from 'react';
import '../resources/button.css';

export default class ResetClassifier extends Component {

  static propTypes = {
    onClick: PropTypes.func.isRequired,
    isSend: PropTypes.bool.isRequired
  };

  _onClick = () => this.props.onClick();

  render() {
    return (
      <div>
        <p> Remove previously saved data: </p>
        <input
          disabled={this.props.isSend}
          className="button"
          type="button"
          value="Reset"
          onClick={() => this._onClick()}
        />
      </div>
    );
  }

}