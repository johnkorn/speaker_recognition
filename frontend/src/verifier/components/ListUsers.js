import React, { Component, PropTypes } from 'react';

export default class ListUsers extends Component {

  static propTypes = {
    users: PropTypes.array.isRequired
  };

  static defaultProps = {
    users: []
  };

  render() {
    return (
      <ul>
        {this.props.users.map(item =>
          <li key={item.key}>{item.value}</li>
        )}
      </ul>
    );
  }

}