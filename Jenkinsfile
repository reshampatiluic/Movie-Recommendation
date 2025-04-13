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
}
