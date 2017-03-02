import 'whatwg-fetch';

export function train(userId, files) {
  const formData = new FormData();
  files.map(_ => formData.append('files_for_training[]', _.data, `${_.key}.wav`));
  formData.append('userid', userId);

  return fetch('/api/train', {
    method: 'POST',
    body: formData
  })
  .then(res => res.json());
}

export const verify = (file) => {
  const formData = new FormData();
  formData.append('file_for_test', file.data, `${file.key}.wav`);

  return fetch('/api/verify',{
    method: 'POST',
    body: formData
  })
    .then(res => res.json());
};

export const reset = () => fetch('/api/reset', {
    method: 'POST'
  })
    .then(res => res.json());

export const getUsers = () => fetch('/api/users')
  .then(res => res.json());