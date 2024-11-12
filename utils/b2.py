        obj = self.bucket.Object(remote_path)
        return obj.get()['Body']

    def file_to_b2(self, local_path, remote_path):
        '''
        Send `local_path` file to `remote_path`.
        '''
        # Guess the type of a file based on its URL
        mimetype, _ = mimetypes.guess_type(local_path)

        if mimetype is None:
            raise Exception("Failed to guess mimetype")
        
        if remote_path in [f.key for f in self.bucket.objects.all()]:
            print(f'Overwriting {remote_path} ...')
        else:
            print(f'Uploading {remote_path} ...')
        
        self.bucket.upload_file(
            Filename=local_path,
            Key=remote_path,
            ExtraArgs={
                "ContentType": mimetype
            }
        )