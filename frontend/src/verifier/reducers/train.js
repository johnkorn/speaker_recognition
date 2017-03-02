
const INITIAL_STATE = {
  send: false,
};

export default (state = INITIAL_STATE, action) => {
  switch (action.type) {
    case 'SEND_TRAIN':
      return {
        ...state,
        send: true
      };
    case 'COMPLETE_TRAIN':
      return {
        ...state,
        send: false
      };
    default:
      return state;
  }
}