pipeline {
    agent any

    stages {
        stage('Setup') {
            steps {
                sh '''
                set -e
                echo Creating virtualenv...
                python3 -m venv .venv

                echo Installing requirements...
                . .venv/bin/activate
                python3 -m pip install --upgrade pip
                python3 -m pip install -r requirements.txt
                '''
            }
        }

        stage('Format') {
            steps {
                sh '''
                set -e
                . .venv/bin/activate
                echo Running linter and checking code format...
                python3 -m black --check .
                '''
            }
        }

        stage('Test') {
            steps {
                sh '''
                set -e
                . .venv/bin/activate
                echo Running tests...
                python3 -m pytest
                '''
            }
        }

<<<<<<< HEAD
        stage('Log Commit Info') {
            steps {
                script {
                    def commitHash = sh(script: "git rev-parse --short HEAD", returnStdout: true).trim()
                    echo "Deploying commit ${commitHash}"

                    // Save commit hash to a file for Docker build
                    writeFile file: 'commit_info.txt', text: "commit: ${commitHash}\ndate: ${new Date()}\n"
                }
            }
        }

        stage('Deploy') {
            steps {
                script {
                    def commitHash = sh(script: "git rev-parse --short HEAD", returnStdout: true).trim()
                    sh """
                    echo Building Docker image with commit ${commitHash}...
                    docker compose build --build-arg GIT_COMMIT=${commitHash}

                    echo Spinning up container...
                    docker compose down --remove-orphans
                    docker compose up -d

                    echo Successfully deployed!
                    """
                }
=======
        stage('Deploy') {
            steps {
                sh '''
                set -e
                echo Building docker image...
                docker compose build

                echo Spinning up container...
                docker compose down --remove-orphans
                docker compose up -d

                echo Successfully deployed!
                '''
>>>>>>> 0b1a69eb1906812fea80cf35c163e016fd45e467
            }
        }
    }

    post {
        always {
            sh '''
            set -e
            . .venv/bin/activate
            echo "Running coverage report..."
            pytest --cov=app tests/
            '''
        }
    }
<<<<<<< HEAD
}
=======
}
>>>>>>> 0b1a69eb1906812fea80cf35c163e016fd45e467
