
import React, { Component, PropTypes } from 'react';
import '../resources/button.css';
import Timer from './Timer';

export default class Verify extends Component {

  static propTypes = {
    speaker: PropTypes.number,
    onClick: PropTypes.func.isRequired,
    isSend: PropTypes.bool.isRequired,
  };

  state = {
    file: null
  };

  _onClick = () => this.props.onClick(this.state.file);

  render() {
    return (
      <div>
        <Timer
          max={1}
          long={5}
          onAdd={(filename, blob) => {
            this.setState({
              file: {
                key: filename,
                data: blob
              }
            });
          }}
          onDelete={(id) => {
            this.setState({
              file: null,
            });
          }}
        />
        <div style={{display: (this.state.file) ? 'initial' : 'none'}}>
          <p> Upload file for testing: </p>
          <p>
            <input
              disabled={this.props.isSend}
              className="button"
              type="button"
              value="Upload"
              onClick={() => this._onClick()}
            />
          </p>
        </div>
        <p> Recognized speaker: </p>
        <p> {this.props.speaker} </p>
      </div>
    );
  }

}