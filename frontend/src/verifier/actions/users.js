
export const send = () => ({ type: 'SEND_USERS' });

export const complete = (error, data) => ({ type: 'COMPLETE_USERS', error, data });

export const add = (key, value) => ({ type: 'ADD_USERS', key, value });

export const update = (key, value) => ({ type: 'UPDATE_USERS', key, value});

export const clear = () => ({ type: 'CLEAR_USERS' });
