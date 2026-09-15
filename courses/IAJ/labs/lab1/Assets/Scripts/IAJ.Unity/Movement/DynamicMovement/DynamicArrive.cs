using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Assets.Scripts.IAJ.Unity.Movement.DynamicMovement
{
    public class DynamicArrive : DynamicVelocityMatch
    {

        public KinematicData ArriveTarget { get; set; }

        public float StopRadius = 1.0f;

        public float SlowRadius = 3.0f;

        public DynamicArrive()
        {
            //this.ArriveTarget = arriveTarget;
            this.ArriveTarget = new KinematicData();
        }

        public override string Name
        {
            get { return "Arrive"; }
        }

        public override MovementOutput GetMovement()
        {
            float desiredSpeed = 0.0f;
            
            //ToDo

            return base.GetMovement();
        }

    }
}
