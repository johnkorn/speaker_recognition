
const INITIAL_STATE = {
  send: false,
};

export default (state = INITIAL_STATE, action) => {
  switch (action.type) {
    case 'SEND_RESET':
      return {
        ...state,
        send: true
      };
    case 'COMPLETE_RESET':
      return {
        ...state,
        send: false
      };
    default:
      return state;
  }
}